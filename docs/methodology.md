# Evaluation protocols and model provenance

## Matched internal OOF comparison

Five outer folds contain disjoint patients from 1,092 eligible development cases.
Clinical transforms use only the corresponding training patients. Age and HR
are converted to numeric, sentinel -1 becomes missing, training-fold median
imputation is followed by training-fold population mean/std scaling, and two
missingness indicators are appended. Encoded inputs use float32.

The compact CNN encodes PRE/EARLY/LATE slices into 64-dimensional embeddings.
Spatial pooling and mean aggregation across slices produce one representation
per patient. Its four convolution blocks use 8, 16, 32 and 64 channels with
BatchNorm, ReLU and max pooling. The dropout/linear patient classifier gives
24,713 trainable parameters. No pretrained backbone is used.

The clinical baseline is logistic regression on four encoded values. Neural
fusion maps these values through a 4→8 clinical branch, concatenates it with the
64 image features and uses a 72→16→1 head. A separate logistic-regression linear
fusion baseline is also compared. The neural heads use a fixed 80-epoch recipe;
the outer fold is not used to choose an epoch. Image embeddings in this OOF
study come from frozen fold-local ordinary CNN checkpoints selected through the
historical inner protocol, **not** from the globally selected EMA checkpoint.

| Approach | OOF ROC-AUC | BCE |
|---|---:|---:|
| Images only | 0.5197 | 0.629661 |
| Clinical only: age + HR | 0.6450 | 0.577562 |
| CNN + clinical: neural fusion | 0.6326 | 0.593395 |
| CNN + clinical: linear fusion | 0.6355 | 0.594180 |

Neural fusion minus clinical AUC is **−0.0124526**, with a conditional paired
95% bootstrap interval **[−0.0377712, 0.0111278]**. It improves numerically over
image-only but does not demonstrate an advantage over clinical-only. Failure to
show improvement is not proof of equivalence or inferiority for every population.

ROC/PR coordinates and loss histories are aggregate exports. No score threshold,
individual probability, patient identifier or label row is published. The loss
plot for neural fusion is the unweighted mean of the five original fold curves.
It is not a training curve for logistic regression and is not a patient-weighted
pooled BCE. Every epoch starts at 1; no cosmetic curve manipulation is applied.

## Historical EMA experiment: a separate protocol

The historical compact CNN used exponential moving averaging with decay 0.8
across successive epoch weights. BatchNorm statistics were recalibrated using
training images only, with dropout disabled. EMA is a checkpoint strategy, not a
different input modality or a replacement for BatchNorm.

Its epoch-15 historical TEST result is **ROC-AUC 0.5523853352**, n=176, with
conditional patient-bootstrap interval [0.4628010, 0.6393734]. This result must not
replace 0.5197 in the matched OOF table: patients, checkpoint lineage and protocol
differ. Development curves use 878 training / 219 validation patients and a
repeatedly reused holdout. There is no per-epoch TEST curve. The TEST had already
been used with other models, so it is not a pristine independent external test.

## OOF clinical predictors versus final multimodal head

The original five clinical logistic regressions had saved coefficients,
transforms and OOF scores but omitted intercepts. Their intercepts were recovered
algebraically, without fitting, in the private preservation phase. Five fresh
processes reproduced 1,092 ordered OOF predictions with maximum absolute error
2.220446049250313e-16, unchanged AUC 0.6450048398806163 and BCE
0.5775619139274957. That private check was repeated after a complete local restore.
Those parameters, records and partition identifiers are **not distributed here**.
The public loader implementation accepts a separately authorized local artifact.

No clinical-only global model trained on all 1,092 patients was created by the
original study; no replacement is fabricated. Averages of fold predictors define
a new strategy and do not automatically inherit the OOF AUC.

The separate final neural-fusion head was fitted on 1,092 development patients
using the historical EMA embedding extractor. It has **no independently measured
performance**. The OOF fusion AUC cannot be assigned directly to that final
checkpoint because OOF used different fold-local image extractors.

