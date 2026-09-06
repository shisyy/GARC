# Node 7.2 Public Data Expansion

## Frozen roster

The preregistration commit `b7a2a3658816fc2f0a93ab046e8091a8542da448`,
file SHA256 `8d00e654a990e2d96b912164bf4c5aaaefc7159304aeceaf3544e0f4ea6d85cc`,
and inventory SHA256 `83bc4abac1a7c9761e7e1f37d4c723e16c30c3ed6cc5409a2f8795dda42c5880`
were verified before selection. The frozen first-36 contains exactly the prior
12-object prefix plus 24 new objects; no object was replaced after QA. The new
set contains 20 Articraft and four NJC assets and is entirely revolute because
that is what the unstratified hash selection produced.

The target-free public manifest SHA256 is
`79c0f575135fed75d92d6837ec78e7aeb7ebd56b19f827d2880639802f4eb61f`.
It contains no target, salt, split rank, score, canonical side, joint limits,
or state fractions.

## Preflight and retained failures

The first roster attempt failed before output because NJC directory assets use
`urdf_sha256`, not `archive_sha256`; the fixed roster was retained. Preflight
v1 exposed an episode-ID parsing bug. V2 fixed that bug and passed all 20
Articraft assets, while all four fixed NJC objects exposed missing published
collision-proxy directories. V3 isolated a read-only mesh-extent compatibility
issue. None of these failures caused a redraw.

The final v4 preflight passed 24/24. For the four NJC objects, a task-owned
visual-only URDF excludes unresolved collision tags while retaining the verified
public visual meshes. This is only a renderer input for RGB-D and segmentation;
no contact, penetration, or physical-ground-truth claim is made. Preflight v4
SHA256 is `a05dcc634cdef8051702d3aec81da9b513c5b27330e87cc7abb433ba0b329a72`.

## Materialization

A 128×128, one-train-plus-one-validation-view-per-state smoke passed 24/24.
The frozen full run then produced 512×512 RGB-A, uint16 millimeter depth, and
uint8 part segmentation for 32 training plus four validation views per state.
The materializer receipt reports 24 ready and zero blocked, SHA256
`97b9df75919c85ce214e22c781624e3a77df864a89d451ee459a9b04f9008c2d`.

The independent public-field audit hashed all 217 files per object, verified
72 frames, a 64/8 train/validation split, 36 frames per state, image dimensions
and dtypes, unique objects, absence of the old 12 outputs, and forbidden-field
absence. All 24 pass. Audit SHA256:
`c44f7756298ca01808f93d677ba2214b087033b8c7a96c7153b7a17f30cb3949`.

## Scratch training

GPU2 and GPU3 were completely idle at launch. Each completed an independent
10-iteration fresh-random smoke and wrote a step-9 checkpoint. Two durable 25k
workers are now running, with 12 unique objects per shard and a new output root
per object. Receipt fields and logs attest that no checkpoint/load/resume
argument was used. GPU4 was not used because its resident process was not owned
by this task.

No score, target, sealed mapping, B_test, or Full22 file was read in this work.

## Atomic queue redistribution

After 14 objects completed, GPU5 became idle. The two original runner parents
were stopped while their already-launched children continued, preventing the
old in-memory queues from claiming another object. A public-only handoff audit
froze the exact eight pending objects, verified that they excluded all completed
and running episodes, and exercised `O_CREAT|O_EXCL` on the destination
filesystem. GPU5 then claimed three objects.

When GPU6 subsequently became fully idle, one still-unclaimed episode was moved
from the GPU2 delta to GPU6 based only on resource availability. The two original
children completed step 24999 naturally; their logs and checkpoints were
verified, original receipts preserved, stale rows superseded, and both stopped
parents terminated. There are no residual stopped processes.

The remaining eight-object queue is now covered exactly once across GPU2/3/5/6
with 2/2/3/1 objects. Every launched episode owns a mode-0600 atomic claim and
starts from random initialization. The queue coverage/receipt-hash audit passed
with SHA256 `30cef6583af23fa38f7d518c0905f7526d25c1e6cbd043b003d75591864fb1e5`.
