# Visual reason labeling and training

The project makes one visual OK/NOK decision and records four supporting reasons.
OCR checks tare and approval markings separately. A reason records visible
evidence; it does not establish why the plant rejected a bottle.

New data and checkpoints use `reason_schema_version: 2`, this exact reason order,
and `project_policy: "primagaz_only_v1"`:

| Field | Present means | Do not infer it from |
| --- | --- | --- |
| `corrosion` | Obvious widespread corrosion-like deterioration matching the agreed examples. | Every rust spot, paint chip, or an unmeasured claim about metal depth. |
| `mechanical_damage` | Clear large deformation or damage to the body or protective structure. | Normal plates, seams, scratches, or unfamiliar construction. |
| `heavy_contamination` | Heavy visible coating, residue or deposits matching the agreed appearance examples. | Every pale fleck, worn paint, or uncertain surface material. |
| `wrong_brand_or_type` | A visibly different brand from PrimaGaz, or an excluded bottle configuration matching the agreed references. | Color alone, an obscured logo, or a guessed model number. |

PrimaGaz ONLY is the project scope. The course's broader ownership example also
permits Benegas; that source remains unchanged. An accepted brand does not establish
that every type is accepted.

## Label each image

Choose **Present**, **Absent** or **Unknown** independently for all four fields.
Present requires visible rejection-level evidence. Absent requires an adequate
view with no visible issue at that level; minor wear can still be visible.
Unknown covers blur, occlusion, incomplete crops and uncertain brand, type or
severity. Add a short note describing the evidence. Several reasons can be present.

Label this view only. Do not copy a defect to other views where it cannot be seen,
derive a reason from an overall NOK label, or fill missing answers with Absent.
Use the private illustrated guide for the agreed examples, including the settled
type and coating decisions. A separate peer calibration worksheet is not required.
Keep source overall labels separate from later owner project decisions when they
differ. Only explicitly reviewed fields receive known targets; the remaining
reason fields stay Unknown. A decision on one pictured image does not label every
image in its group or establish a universal model or color rule.

## Keep splits and references separate

Use separate private Label Studio projects for TRAIN, VALIDATION and FINAL TEST.
One task is one image. Primary video cameras are `front`,
`back_left` and `back_right`; top views are excluded. Keep every known physical
bottle group in one split. Do not inspect final-test predictions or adapt the
rubric from final-test failures.

Quarantined source-identity flags remain traceable but cannot train. A recorded
passage is not proof of a unique physical cylinder. Verify independent overall
OK/NOK reference coverage before claiming bottle accuracy. Stored task counts
are not counts of reviewed labels or usable positive bottles; exact private
counts and exclusions belong in the dataset receipt.

Source visual OK/NOK, owner project OK/NOK, visible reasons and plant
`Primagaz_status` have different provenance. Keep all of them. A plant-OK-derived
Absent default is not an individual reason review.

## Prepare and train

Export each project as Label Studio JSON. Prepare a fresh private folder with the
matching crosswalk:

```bash
python src/scripts/prepare_reason_dataset.py --exports train.json validation.json final-test.json --crosswalk private-crosswalk.csv --output data/prepared/visual-reasons-v2
python src/scripts/train_classifier.py train --data data/prepared/visual-reasons-v2 --output src/runs/classifier/reasons-v2
```

The exporter checks project and split identity, group membership, image hashes,
four required answers and annotation validity. Unknown becomes `null` with mask
0. Unannotated rows remain context with no supervision. Quarantined rows remain
traceable but are not approved training examples. Preserve the export receipt.

Training fits approved TRAIN images with at least one known enabled target.
These may be photos or video crops. Validation loss, checkpoint selection and
reason cutoffs use complete video triplets from the three primary cameras.
Three unrelated photos are not a video triplet. A reason needs positive and
negative support in TRAIN and in eligible validation bottle truth before it can
be enabled. Additional labeled photos help fitting but do not substitute for
validation triplet coverage.

The frozen MobileNet encoder turns each crop into features once. A binary head
scores NOK and a separate head scores reasons. Binary inference combines all
three NOK scores by configurable maximum or mean. Each reason uses its own
maximum across the three views and its saved cutoff. Supporting reasons never
override the independent binary decision. Report unresolved and disagreeing
records honestly, then combine OCR only under explicit overall inspection rules.

## Preserve historical meaning

Schema 1 used `major_damage`, `heavy_dirt`, `wrong_brand_or_model`. Old damage
positives included corrosion. Keep their original names in inference and old
results; do not copy the old damage weights or labels into mechanical damage.
The old dirt output lacked accepted positive supervision. New training accepts
schema 2 only; inference and evaluation can still read recognized legacy models
with their original schema. Dataset and checkpoint schemas must agree.

No new trained or validated four-reason checkpoint follows merely from changing
the form and code. A few reviewed positive images do not supply validation
support for four outputs. Historical binary weights also lack a confirmed
`primagaz_only_v1` policy; loading new reason weights does not change their learned
decision. Refit binary weights using reviewed current-policy visual references
and preserve the reserved recording and its photo counterparts for final evaluation before
claiming current-policy or unseen-recording accuracy.
