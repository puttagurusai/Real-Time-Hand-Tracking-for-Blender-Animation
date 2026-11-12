"""Decide which rig each detected hand drives (pure Python, no camera needed).

Slot 0 is the person's LEFT hand (the rig "Armature", at the origin) and slot 1 their RIGHT hand (the rig
"Armature.001", on the +X side), so the right hand drives the rig on the right of Blender's front view.
MediaPipe labels every detected hand "Right" or "Left"; the sender flips the picture like a mirror,
which is the way MediaPipe expects it, so the label is the person's real hand. The label is not
perfect (it flickers, and a hand seen from the back is sometimes called the wrong one), so:

  * a new hand takes the slot its label names; with no usable label, the side of the picture
    decides (in a mirrored picture the right hand is on the right);
  * after that the hand keeps its slot by where it is: it follows the nearest remembered position,
    allowing for its motion, so a flickering label cannot make two hands swap;
  * only when a hand's label has said the opposite for SWITCH_FRAMES frames in a row does it move
    to the other slot (or swap with the other hand if that one disagrees too).

A slot remembers its hand for HOLD_FRAMES frames after it disappeared, so a hand that returns
nearby keeps its slot.
"""
LEFT, RIGHT = 0, 1
SLOT_NAMES = ("left", "right")
_WANTED = {"Right": RIGHT, "Left": LEFT}


class SlotAssigner:
    def __init__(self, slots=2, max_jump=0.3, hold_frames=15, switch_frames=8):
        """max_jump: farthest a hand may move between frames, as a fraction of the image width (after
        accounting for its recent motion). hold_frames: how long a slot remembers an absent hand.
        switch_frames: how many frames in a row a label must disagree before the hand changes slot."""
        self.slots = slots
        self.max_jump = max_jump
        self.hold_frames = hold_frames
        self.switch_frames = switch_frames
        self.reset()

    def reset(self):
        self._last = {}    # slot -> (x, y) where its hand was last seen
        self._vel = {}     # slot -> (dx, dy) its last frame-to-frame movement
        self._age = {}     # slot -> frames since it was seen
        self._bad = {}     # slot -> frames in a row its hand's label named the other hand

    def _predicted(self, slot):
        (x, y), (dx, dy) = self._last[slot], self._vel.get(slot, (0.0, 0.0))
        return x + dx, y + dy

    def _preferred(self, label, position, taken):
        """The slot a new hand should take: its label's, else the one on its side of the picture."""
        want = _WANTED.get(label)
        if want is not None and want < self.slots and want not in taken:
            return want
        free = [s for s in range(self.slots) if s not in taken]
        if not free:
            return None
        side = RIGHT if position[0] >= 0.5 else LEFT  # mirrored picture: the right hand is on the right
        return side if side in free else free[0]

    def assign(self, positions, labels=None):
        """positions: [(x, y)] image position of each detected hand's wrist, both in 0..1; labels: the
        matching MediaPipe labels ("Right", "Left" or anything else for none). Returns
        {slot: index into positions} for the hands that got a slot."""
        labels = list(labels) if labels is not None else [None] * len(positions)
        count = len(positions)
        order = sorted(range(count), key=lambda i: positions[i][0])[:self.slots] if count > self.slots \
            else list(range(count))
        result, taken = {}, set()

        # 1. a hand that was here a moment ago stays in its slot
        pairs = []
        for i in order:
            for slot in self._last:
                px, py = self._predicted(slot)
                d = ((positions[i][0] - px) ** 2 + (positions[i][1] - py) ** 2) ** 0.5
                if d <= self.max_jump:
                    pairs.append((d, i, slot))
        for d, i, slot in sorted(pairs):
            if i not in taken and slot not in result:
                result[slot] = i
                taken.add(i)

        # 2. a new hand takes the slot of its label, else of its side (the rightmost new hand first)
        for i in sorted((i for i in order if i not in taken), key=lambda i: -positions[i][0]):
            slot = self._preferred(labels[i], positions[i], set(result))
            if slot is None:
                break
            result[slot] = i
            taken.add(i)

        # 3. a label that keeps naming the other hand moves the hand there (or swaps the two)
        disagree = {s: _WANTED.get(labels[i]) not in (None, s) for s, i in result.items()
                    if _WANTED.get(labels[i]) is not None}
        for slot in list(result):
            if slot in disagree:
                self._bad[slot] = self._bad.get(slot, 0) + 1 if disagree[slot] else 0
        for slot in sorted(result):
            if slot not in result or self._bad.get(slot, 0) < self.switch_frames or self.slots < 2:
                continue
            other = LEFT if slot == RIGHT else RIGHT
            if other not in result:
                result[other] = result.pop(slot)
                self._move(slot, other)
            elif self._bad.get(other, 0) >= self.switch_frames // 2:
                result[slot], result[other] = result[other], result[slot]
                self._swap(slot, other)
            else:
                continue
            self._bad[slot] = self._bad[other] = 0

        # 4. remember where everyone is
        for slot in range(self.slots):
            if slot in result:
                x, y = positions[result[slot]]
                self._vel[slot] = ((x - self._last[slot][0], y - self._last[slot][1])
                                   if slot in self._last else (0.0, 0.0))
                self._last[slot], self._age[slot] = (x, y), 0
            elif slot in self._last:
                self._age[slot] += 1
                if self._age[slot] > self.hold_frames:
                    for memory in (self._last, self._age, self._vel, self._bad):
                        memory.pop(slot, None)
        return result

    def _move(self, old, new):
        for memory in (self._last, self._age, self._vel):
            if old in memory:
                memory[new] = memory.pop(old)

    def _swap(self, a, b):
        for memory in (self._last, self._age, self._vel):
            if a in memory and b in memory:
                memory[a], memory[b] = memory[b], memory[a]
