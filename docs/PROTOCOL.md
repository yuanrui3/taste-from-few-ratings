# Appearance taste model: protocol

> **About this file.** This is the lab notebook of the first study: every test's decision rule was written
> here before the test ran, then its result was added. Quoted instructions are the rater's own. No person in
> the photo pool is named anywhere; photos, ratings and account lists were never published.


Written 2026-11-12, before any account was collected or rated. Changes after this date are
listed at the end with their reason; the pass rule may not be changed once ratings exist.

Scripts named below live in this folder (the pipeline) or in `experiments/` (one-off studies,
moved there on 2026-11-13). How to run them: `README.md`.

**What is new here, in short** (details and numbers in `README.md`, section "What is new"):
one rater's taste for whole people, not a crowd's score for cropped faces; learned from ordinary
social-media photos with scenery removed and clothing greyed out, and shown by control tests not
to depend on scenery, clothing or photo style; usable from about 20 ratings, with the people to
rate chosen without any ratings and a crowd face model carried over from studio photos through
the shared latent space; checked on 60 other raters; every rule written down before its test.

## Goal

Predict the user's 0-10 rating of a person's appearance from that person's Instagram photos.

- Counts: face, body, posture, skin tone, hair, small accessories (earrings and the like).
- Counts a little: clothing style. It changes across a profile, so it should have minimal impact.
- Does not count: scenery, vibe, activities, captions, follower numbers.

One account is one person and gets one rating.

## Accounts

- Every public account the user follows that shows one person (decision 2026-11-12: "All public
  accounts"). Private accounts are left out.
- About one third more accounts the user does not follow, from Instagram's "similar accounts"
  suggestions for followed accounts, so that ratings cover the whole scale (decision 2026-11-12).
- An account is used only if its owner can be found (next section). Brands, meme pages and group
  accounts drop out there, before any rating is seen.

## Photos and the owner

1. Up to 24 recent images per account: photos, carousel images, reel cover images. Downloaded
   straight to the cluster's private project folder, never to the rater's computer,
   whose disk is nearly full. The Mac keeps only small thumbnails for the rating page.
2. Owner = the face most of the account's photos agree on (SFace, same person at cosine >= 0.363;
   the agreement idea of `tools/select_frames.py`). At least 4 photos must show the owner's face,
   otherwise the account is dropped. Other people in a photo are blanked out.
3. Cut-out: the clothes parser of `idscore/parsing.py` (SegFormer) separates person, clothing and
   background. The background becomes flat gray.

## Rating

- A local page shows about 12 cut-outs of the owner on gray, without the handle. The user gives a
  whole number 0-10, or "skip" (not one person / cannot judge). Accounts come in random order.
- About 10% of accounts come back later in the session, unannounced, to measure how consistent the
  user's own ratings are. That consistency is the ceiling any model can reach.

## Split (frozen before rating)

Accounts that pass the owner step are split at random (fixed seed) into 75% training and 25% test,
stratified by followed / not followed. The split file is written, and its SHA-256 recorded below,
before the rating page opens. Model choices use training accounts only (5-fold cross-validation).
The test accounts are scored once, at the end, and the result is reported whatever it is.

## Model

No step before the last one sees ratings ("unsupervised").

1. Open-source image models embed each photo: DINOv2 (Apache-2.0, self-supervised) and OpenCLIP
   (MIT), on the cut-out, on the face crop, and on the cut-out with clothing turned flat gray.
   Plus the SFace face embedding, skin colour from the parser's skin pixels, and body proportions
   from pose keypoints where the whole body shows.
2. Body shape from a separate body-model pipeline (Multi-HMR) is added only if it improves
   cross-validated accuracy on the training accounts.
3. Outfit- and scene-invariance without ratings: a small projection trained on all collected photos
   so that photos of the same person land together whatever they wear and wherever they are
   (contrastive learning, positives = same account).
4. Account vector = average over the owner's photos. A ridge-type regression maps it to the rating;
   which embeddings and settings are used is decided by cross-validation on the training accounts.

### Explicit features (added 2026-11-12 at the user's suggestion, before any data)

A small "less machine learning" part for features many people prefer directly. Each is measured
per photo and averaged per account, without ratings:

- Hair colour: colour of the parser's hair pixels (lightness, hue), so blonde / brown / black / red
  falls out as numbers. Dye and lighting change it, so it is averaged over the profile.
- Leanness: a rough body-fat category (lean / average / higher), not a percentage. Instagram photos
  have flattering poses, light and editing, so a percentage would be false precision.
- Height: only where a bio or caption states it. Photos alone do not give height (no scale, heels,
  camera angle); estimates from photos are not used.
- Also cheap: skin tone (already above), visible muscle definition, hair length.

Used two ways: (a) a transparent model on these features alone, and (b) these features added to the
embedding model, kept only if cross-validation on the training accounts says they help. The user may
also state preferences up front (e.g. preferred hair colour, build); a score built from those stated
rules is reported next to the learned model ("what you say vs what you rate"). None of this changes
the pass rule. Per-person estimates of body fat or other body traits are internal features only and
are never listed or shown per person.

### Stated dealbreakers (2026-11-12, not used)

After about 45 ratings the user named three things they reject outright (failed plastic surgery,
overly large fake breasts/buttocks, large tattoo coverage). A detector-and-veto add-on and rating
tags were proposed; the user declined ("I don't need it written down I will ignore those"), so the
tags were removed from the page after rating ~52 and no veto rule is used. Their ratings carry it.

## Few-rating objective (added 2026-11-12 at the user's request, before any rating)

The user: "I want the model to be able to predict a person's taste with minimum number of
reviews", possibly via a set of stock images that most truthfully and effectively reveal a
person's taste, taken from this dataset.

- Probe set: K accounts from the training split, chosen by a rule that uses no ratings, only the
  appearance features (greedy coverage of the embedding space: k-center / D-optimal design), so it
  is not tuned to this user's answers and could be shown to anyone.
- Learning curve on the frozen test accounts: accuracy (Spearman) when the model is trained on the
  first n ratings, n = 5, 10, 15, 20, 30, 40, 60, 80, all; for (a) the probe set in its fixed
  order, (b) random training accounts (mean and spread over 50 random orders), (c) active
  selection (next account = the one the current model is least sure about).
- Reported claim: the smallest n at which the probe set reaches 90% of the all-ratings accuracy,
  and whether the probe set beats random at equal n (bootstrap interval of the difference).
- Limit: with one rater this shows the probe set works for this user only; showing it works for
  people in general needs a few more raters.

## Pass rule (fixed now)

On the test accounts, all three must hold:

1. Spearman correlation between predicted and actual ratings at least 0.50, with the lower end of
   its 95% bootstrap interval at least 0.30.
2. Better than crowd taste: beats a generic open-source beauty/aesthetic score (LAION aesthetic
   predictor on the same cut-outs); the 95% bootstrap interval of the difference lies above 0.
3. Appearance, not scenery or fashion: beats two controls built the same way, one that sees only the
   background (person blanked) and one that sees only the clothing; both differences' 95% bootstrap
   intervals lie above 0.

Also reported, not part of the rule: correlation as a share of the user's own re-rating consistency,
how often the model orders two accounts correctly when their ratings differ by 2 or more, mean
absolute error, how many of the top 10% predicted the user rated 8 or more, and results for women
and men separately.

## Privacy and Instagram

- Photos, cut-outs, features and ratings stay on the rater's computer and the cluster's private folder. They are never
  committed, published or shared. No avatars or renders of these people. Deleted on request.
- The user signs in themselves in the app's browser. Browsing is read-only (no likes, follows or
  messages), slow (at least 15 s between profiles, randomised) and stops at any warning or
  challenge. Automated browsing goes against Instagram's terms; the account could be limited.
- Tracked files name no accounts.

## Split record

Frozen 2026-11-12, before any rating, by `freeze_split.py` (seed 20261113) on the 412 accounts the
owner step kept (55 of 467 dropped: fewer than 4 photos agree on one face).
- Train 309, test 103 (33 of the test accounts are followed by the user).
- Rating sequence 453 = 412 accounts + 41 unannounced repeats.
- `split.json` SHA-256: `c905e094e5755fbe96f30a5d701b7e7240e81bd144d993a050b1b1d82b5c5a7b`

Same-person fix (2026-11-12, after about 60 ratings; no rating used, test set only made
stricter): some people have two accounts in the pool. Rule: two accounts are the same person if
the share of their cross-account photo pairs with SFace cosine >= 0.363 is at least 0.9 times the
lower within-account share. Checked on all pairs of the 412 accounts; 7 pairs pass, with a clear
gap to the next (0.82). The 3 test accounts whose twin is in training leave the test set
(names in the private `split/same_person.json`): test 100 accounts. Same-person pairs stay
together in cross-validation folds. Record: `split/same_person.json`. Second-pass (repeat)
ratings are used only to measure the user's consistency; training and testing use first-pass
ratings.

## Results (2026-11-12; test run once, job 12345049)

Ratings: 407 rated + 5 skipped first-pass; repeats n=40, the user's own consistency Spearman 0.869
(mean absolute difference 0.35). Training 305 rated accounts, test 99.

Model chosen by training CV only (`train_eval.py --stage cv`, job 12344994): OpenCLIP ViT-L/14 on
the cut-out with clothing grayed + the face crop, account means, ridge (alpha 3000); CV 0.595.
DINOv2 was weak (0.21); explicit features alone 0.24 and did not add; the rating-free
outfit-invariant projection matched the winner (0.59) with 128 dims.

Test (pass rule): Spearman 0.668 [0.530, 0.773] -> (1) pass; crowd aesthetic score -0.054,
difference [0.511, 0.917] -> (2) pass; scene-only control 0.336, difference [0.126, 0.537];
clothing-only control 0.265, difference [0.235, 0.571] -> (3) pass. **PASS.**
Also: 77% of the user's own consistency; correct order for 85% of pairs rated 2+ apart; mean
absolute error 0.89; of the top 10% predicted, 33% were rated 8+ (base rate 16%); women 0.70
(n=91), men 0.61 (n=8, too few to read).

Few-rating curve (test Spearman by number of training ratings): random 0.15 (10), 0.31 (20),
0.42 (40), 0.52 (80), 0.56 (120), 0.60 (160), 0.67 (all 305). 90% of the all-ratings accuracy
needs about 160 random ratings. The farthest-first probe set did NOT beat random (0.34 vs 0.42
at 40; 0.41 vs 0.52 at 80), nor did active selection. Farthest-first picks unusual accounts first,
which a linear model generalises badly from. Any better probe set must now be chosen and checked
on training data only; this test set has been used.

## Probe-set search (planned 2026-11-12, after the test; training data only)

Goal: fewer ratings for the same accuracy. Evaluated only on the 305 rated training accounts:
20 random splits (same-person accounts kept together) into a pool (70%) and an evaluation part
(30%). A method picks n accounts from the pool without seeing ratings, a model is fitted on
those n ratings, and Spearman is measured on the evaluation part. n = 5, 10, 20, 30, 40, 60, 80.

- Selection: random (reference); farthest-first (the failed one); k-means medoids (the account
  nearest each of n cluster centres); kernel herding (greedy, matches the pool's distribution).
- Model given n ratings: ridge on the full winner features, or on a low-dimensional summary
  (PCA of all 412 accounts' features, k = 8, 16, 32; no ratings used), with the ridge strength
  chosen by leave-one-out on the n ratings themselves.
- Zero-rating prior (extra arm): a classifier trained on the pool's follow status (followed vs
  Instagram-suggested; the user's follow list, not ratings) scores everyone; used alone (0
  ratings) and as an extra input next to the n ratings.
- Decision rule (fixed now): the best method is the one with the highest mean evaluation
  Spearman averaged over n = 10, 20, 40. It counts as better than random only if it beats random
  in at least 15 of the 20 splits at n = 20. The test set is then looked at once more for that
  method only, labelled as a second look (not a clean test).

### Probe-set search: results (job 12345189, `results/probe_search.json`)

Mean evaluation Spearman over 20 splits (training accounts only):

| method | n=5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|
| plain random (old model) | 0.15 | 0.21 | 0.31 | 0.40 | 0.51 |
| random + follow-list prior | 0.29 | 0.38 | 0.44 | 0.49 | 0.55 |
| **kernel herding + follow-list prior (rule's pick)** | 0.38 | 0.44 | 0.48 | 0.53 | 0.56 |
| k-means medoids + prior | 0.17 | 0.39 | 0.48 | 0.51 | 0.55 |
| farthest-first + prior | 0.39 | 0.40 | 0.41 | 0.48 | 0.56 |
| follow-list prior alone, 0 ratings | 0.50 | | | | |

- Rule: herding + full features + prior wins; it beats plain random at n=20 in 20/20 splits, but
  random selection with the same model in only 12/20. Most of the gain is the follow-list prior;
  the selection itself helps mainly at 5-10 ratings.
- The way the prior and the ratings are combined is poor: below about 40-60 ratings the
  combination scores lower than the prior alone (0.50).
- Recognition check: within never-followed accounts only, the prior still reaches 0.32 (followed
  0.13), so it carries appearance taste, not only "accounts I know". For comparison the full model
  on the test set: 0.71 within never-followed (n=67), 0.12 within followed (n=32).
- Second look at the used test set (herding + prior): 0 ratings 0.50; 10: 0.32; 20: 0.37;
  40: 0.42; 80: 0.54. Same picture: the prior alone is the best few-rating predictor so far.
- Probe order (rating-free, first 40): `results/probe_set.json`.

### Combining the follow-list prior with few ratings (planned 2026-11-12, before running)

Same 20 training splits, same evaluation. The prior z = the follow classifier's score
(standardized on the pool). Ratings come from kernel-herding picks (and random, for reference).
Candidates, each keeping the prior as the starting point:

- Transfer ridge: fit ridge to y - c*z, add c*z back; c in {0, .25, .5, .75, 1, 1.5} (rating
  points per prior SD) and the ridge strength chosen together by leave-one-out on the n ratings.
- Fixed offset: c = r0 * SD(y) with r0 in {0.3, 0.5} fixed in advance; ridge on the residual.
- Rank blend: (1 - w) * rank(prior) + w * rank(ratings-only model), w = n / (n + k),
  k in {10, 20, 40, 80} fixed in advance.

Rule (fixed now): best = highest mean evaluation Spearman averaged over n = 5, 10, 20, 40. It
"fixes the problem" only if it beats the prior alone at n = 20 in at least 15 of the 20 splits.
Then one second look at the used test set for that method only.

Results (job 12345329, `results/combine_search.json`): prior alone 0.498. Rule's pick: kernel
herding + rank blend k=40: 0.505 (5 ratings), 0.513 (10), 0.505 (20), 0.540 (40), 0.568 (80);
ratings-only with the same picks 0.20 / 0.28 / 0.37 / 0.48 / 0.54. It beats the prior alone at
n=20 in only 9/20 splits, so by the rule it does NOT fix the problem: the combination no longer
falls below the prior, but a handful of ratings adds nothing measurable on top of the follow
list; ratings start to add from about 40. Second look at the used test set: prior 0.499; 5: 0.478,
10: 0.464, 20: 0.484, 40: 0.497, 80: 0.526.

## What drives the taste (exploratory, training accounts only; `taste_explain.py`, `taste_latent.py`)

- OpenCLIP text probes (attribute vs its opposite, Spearman with ratings, all / strangers only):
  slim vs curvy figure +0.30 / +0.21; lip fillers -0.25 / -0.24; heavy makeup -0.24 / -0.30; wide
  hips -0.23 / -0.16; full lips -0.23 / -0.23; large eyes -0.21; sharp jawline -0.21; natural vs
  glamorous look +0.14 / +0.20; dark vs light skin -0.12 / -0.16; visible tattoos -0.06 / -0.18;
  athletic, tan, smile, hair length about 0. Single attributes are weak; the model uses many.
- Looking at the top and bottom 16: top-rated are mostly slim fashion or fitness physiques in
  editorial, full-body shots; bottom-rated include casual selfies and stream frames, glam/contour
  looks and some very curvy silhouettes. Possible confound: professional vs casual photography
  is partly visible in the cut-outs (the scene-only control reached 0.34 on the test).
- Latent types (k-means, no ratings): mean rating per type 5.4-6.7; the editorial-model type
  is highest, the glam-influencer type lowest. Taste is a direction within types more than a type.
- Network (metadata, not used in the final model per the scope decision below): image +
  network 0.69 vs image 0.60 in training CV, but among strangers 0.49 vs 0.51, so the gain is
  only from follow status.

## Scope decision (user, 2026-11-12)

"this project still needs to stay on the computer vision trajectory, ideally our final model
should not rely on any Instagram metadata". The final model uses images only (as the tested
model already does). The follow-list prior and the network features (follows, similar-accounts
lists) were exploratory and are not part of it; few-rating work continues on images alone
(vision-only figures: kernel-herding picks, ratings-only model, 0.20 / 0.28 / 0.37 / 0.48 / 0.54
at 5 / 10 / 20 / 40 / 80 ratings).

## Vision-only few-rating: attribute basis (planned 2026-11-12, before running)

Idea: with few ratings, learn a weight for each of a few describable attributes instead of for
1536 latent dimensions. Images only.

- Basis, two versions: (main) all 31 OpenCLIP text-probe attributes of `taste_explain.py`, since
  the user noted that excluded ones "may matter to other audience" and the basis should work for
  anyone, with each person's ratings deciding the weights; (personal) the 27 left after removing
  what the user said does not matter to them ("I actually don't care about eye size, hair color,
  freckles"). Scores per account as before. The rule below is applied to the main version.
- Same 20 training splits, n = 5, 10, 20, 40, 80 ratings, kernel-herding picks (PCA-32 of the
  latent codes) and random picks. Candidates: ratings-only on the full latent codes (the current
  vision-only reference); ridge on the 27 attributes; attributes first then a ridge on the full
  codes for what is left; attributes and full codes side by side (attributes weighted up x5);
  rank blend of the attribute model and the full model, w = n / (n + 40).
- Rule (fixed now): best = highest mean evaluation Spearman over n = 5, 10, 20, 40. It improves
  on the reference only if it beats herding + ratings-only full codes at n = 20 in at least 15 of
  the 20 splits. Then one second look at the used test set for that method.

Results (job 12345733, `results/attr_fewshot.json`): the attribute basis does NOT help. The rule's
pick is the reference itself (herding + full latent codes: 0.20 / 0.28 / 0.37 / 0.48 / 0.54 at
5 / 10 / 20 / 40 / 80). Attributes alone: 0.19 / 0.20 / 0.23 / 0.33 / 0.35, and only 0.41 with all
pool ratings (full codes 0.60). Adding attributes to the codes gains 0.02 at 5 ratings and loses
from 20 on. Main (31) and personal (27) bases behave the same. Reading: the 31 describable
attributes hold much less of the taste than the latent codes; most of it is not captured by these
words. Second look at the used test set for the reference: 5: -0.09, 10: 0.02, 20: 0.27, 40: 0.35,
80: 0.50.

## Fitness score (planned 2026-11-13, before running; images only)

The user: "one thing I care about is how fit they are"; asked whether muscle detection can tell.

- Crops from the owner's body keypoints (Keypoint R-CNN, already computed): lower torso (abs),
  upper arms (shoulder to elbow), thighs (hip to knee), from the original photo.
- Each crop: is the body part bare? (OpenCLIP: "a bare stomach" vs "a stomach covered by a
  shirt", and the same for arms and legs); and how defined is it? ("a lean toned stomach with
  visible abs" vs "a soft stomach with some belly fat"; "toned muscular arms" vs "soft untoned
  arms"; "toned athletic thighs" vs "soft untoned thighs"). Only crops judged bare count,
  weighted by that judgement.
- Per person: the weighted mean definition per region, standardized across people, averaged over
  the regions they show = fitness score; people with no bare region get none.
- Check without ratings: sheets of the 12 highest and 12 lowest scored people; the score is only
  used further if those sheets plainly run from fit to not fit.
- Then, training accounts only: Spearman of the score with the ratings (all, strangers only), and
  whether adding it to the main model raises cross-validated Spearman (5 folds x 3 repeats).

Results (job 12348923, `results/fitness.json`): 12,302 crops scored (abs 2,556, arms 5,958, thighs
3,788); 396 of 412 people scored (360 on 2+ regions). Sheet check: the top 12 are plainly very
muscular (male and female bodybuilders, defined physiques), with two errors (a poster, a suited
man); the bottom 12 are mostly slim women in everyday clothes plus babies and close-ups. So the
score measures visible muscularity, not fitness in the sense of a lean body; slim fit women without
visible definition score low. Against the ratings: 0.03 overall, 0.03 strangers only; women only
0.04 (n=277; abs 0.04, arms 0.06, thighs 0.00); men 0.41 (n=13, too few). Adding it to the main
model: CV 0.5946 -> 0.5948 (no change). Reading: for the women rated, visible muscle does not
drive the ratings; leanness does, through "slim vs curvy figure" (+0.30). Not added to the model.
Women-only rescoring (user: "what if you exclude men from the equation"; standardized among the
386 women): top 12 are fitness competitors and visibly defined women, bottom 12 slim women in
everyday clothes, so within women it ranks visible definition. Ratings: 0.04 (n=277), strangers
0.03 (n=183), 2+ regions 0.03; mean rating by score fifth 5.9 / 6.2 / 5.9 / 5.8 / 6.3, flat; the
12 highest averaged 6.3, the 12 lowest 6.0. `results/fitness_women.json`.

## Body shape: skipped (user, 2026-11-13)

The plan's step "body shape from a body-model pipeline (Multi-HMR), kept only if it helps" was
not run. Its weights and body-model files were not available to this project; the expected gain was near zero (shape estimates sit close to the average body, the image
codes already capture build, keypoint proportions and describable attributes added nothing). The
user decided: "skip then". With this, the original model plan is complete.

## Overnight session (user, 2026-11-13 01:43: "full authorization to approach our high-level
## objective until 8:30am today"). Images only, training accounts only, test set not used again.

### Photo-style confound (planned before running; `style_confound.py`)

Style score per photo: OpenCLIP probes on the cut-out the model sees ("a professional studio
photoshoot" vs "a casual amateur phone photo"; "a professionally retouched photo" vs "an unedited
snapshot"; "a fashion magazine photo" vs "a selfie"), standardized and averaged.
1. Spearman of the account's mean style with the ratings (all, strangers only).
2. Partial Spearman of the main model's cross-validated predictions with the ratings, holding style
   fixed.
3. De-styled model: the three style directions removed from every photo code before averaging;
   cross-validated Spearman compared with the main model (0.595).
4. Within-person leak: the main model applied to single photos; Spearman between a photo's
   prediction and its style score within each person, averaged over people with 6+ photos.
Rule (fixed now): the confound is material if the de-styled model loses 0.05 or more, or the
within-person leak is 0.20 or more. If material and the de-styled model loses less than 0.03, it
becomes the recommended model.

Results (job 12349557, `results/style_confound.json`): NOT material. Style vs rating 0.09 (strangers
0.07); de-styled model 0.595 vs main 0.595; main model holding style fixed 0.591; within-person
leak 0.09 (264 people). The model does not reward professional photography; the earlier visual
impression (top-rated people often in editorial shots) reflects who they are, not how the
photos were taken. Main model unchanged.

### Larger image models (planned before running; `embed_more.py`, `backbone_eval.py`)

Same inputs (no-clothing cut-out + face crop, account means), embedded with OpenCLIP ViT-H-14
(laion2b) and SigLIP SO400M-384 (webli), alone and concatenated with the current OpenCLIP
ViT-L-14. Training accounts only:
- Full-data: 5-fold x 3 grouped CV, ridge strength by CV. A model replaces the current one only if
  its CV Spearman is at least 0.02 higher (current 0.595).
- Few-rating: the same 20 pool/evaluation splits, kernel-herding picks on each model's own
  PCA-32, n = 5, 10, 20, 40, 80, ridge by leave-one-out. Best = highest mean over n = 5, 10, 20,
  40; it counts as better only if it beats the current model at n = 20 in at least 15 of 20 splits.

Results (jobs 12350573 embedding, 12350798 comparison; `results/backbone_eval.json`):
Full-data CV: OpenCLIP L-14 0.595; ViT-H-14 0.603; SigLIP 0.635; L+H 0.611; L+SigLIP 0.636;
H+SigLIP 0.631; all three 0.628. -> by the rule the full model becomes **OpenCLIP L-14 + SigLIP**
(+0.041). Few-rating (herding): L-14 stays best (0.20 / 0.28 / 0.37 / 0.48 / 0.54); SigLIP alone
0.13 / 0.19 / 0.32 / 0.41 / 0.54 -> no model beats L-14 at n = 20 (0/20). Larger codes pay off only
with many ratings. Next, planned now: (1) one second look at the used test set for L + SigLIP,
labelled as such (the clean test result stays 0.668 for the original model); (2) crowd face
prior rebuilt with SigLIP and ViT-H embeddings of the SCUT faces and the user's face crops,
compared by zero-rating Spearman on training accounts; the best feeds the k = 20 blend.
Second look (job 12350881, `results/followups.json`; NOT a clean test, the test set was used
once already): OpenCLIP L-14 + SigLIP, ridge alpha 3000 as before, 0.716 [0.58, 0.81] on the 99
test accounts (clean original result 0.668 [0.53, 0.77]).
Crowd prior with other embeddings (same job): zero-rating Spearman with the user, training
accounts: L-14 0.350 (strangers 0.336); ViT-H 0.324 (0.332); SigLIP 0.277 (0.296); L-14 + SigLIP
0.335; all three 0.337 -> the L-14 crowd prior stays (blend k = 20 unchanged: 0.37 / 0.41 / 0.46 /
0.53 / 0.57). SCUT embeddings for the other models saved in `datasets/scut/scut_more.npz`.

### Many raters: SCUT-FBP5500 (planned before running; `scut_prep.py`, `scut_eval.py`)

Public research dataset (non-commercial research use): 5,500 face photos (Asian and Caucasian,
women and men), each rated 1-5 by the same 60 volunteers; `All_Ratings.xlsx` holds every rater's
own scores. Downloaded to the cluster's `datasets/scut/` only. Faces embedded with the same OpenCLIP
ViT-L-14 as the main model's face input.
A. Few-rating test across 60 people: for each rater, 5 random splits of the images into pool (70%)
   and evaluation (30%); n = 5, 10, 20, 40, 80 ratings picked by random or kernel herding (PCA-32);
   models: ratings only (ridge, leave-one-out strength); crowd prior only (a ridge trained on the
   mean score of the OTHER 59 raters, pool images only); prior + ratings (rank blend,
   w = n / (n + 40), and transfer ridge as in "Combining..."). Measure: Spearman with that rater's
   own scores on the evaluation images. Also the same against the rater's deviation from the crowd
   (own score minus the others' mean), i.e. the personal part of taste.
   Rules (fixed now): herding is better than random if it wins at n = 20 for at least 45 of the 60
   raters; the crowd prior helps if prior + ratings beats ratings only at n = 20 for at least 45.
B. Transfer to this user: the crowd face-beauty model (all 60 raters' means, all 5,500 faces)
   applied to the user's accounts' face crops: Spearman with the user's ratings (training accounts,
   all and strangers). If at least 0.20, test prior + few ratings on the user's 20 training splits:
   it helps if it beats ratings only at n = 20 in at least 15 of 20 splits.

Results (job 12349827, `results/scut_eval.json`):
A. 60 raters. Taste in SCUT is mostly shared: each rater vs the others' mean, median Spearman
   0.78; the crowd prior alone predicts a rater's own scores at 0.72 with zero ratings. Ratings
   only: herding 0.58 / 0.62 / 0.66 at 20 / 40 / 80 (random 0.53 / 0.60 / 0.64). Herding beats
   random at n = 20 for 58/60 raters -> the probe method generalises beyond this one user.
   Prior + ratings beats ratings only at n = 20 for 60/60 raters (rank blend and transfer ridge)
   -> the crowd prior helps. The personal part (own minus crowd) stays unpredictable (about
   0.00-0.03 up to 80 ratings). Some n = 5 / 10 cells are NaN (constant predictions for a rater).
B. Transfer to the user (images and public data only): crowd face beauty predicts the user's
   ratings at 0.350 (strangers 0.336) with zero ratings (LAION aesthetic: -0.05). Prior + ratings
   (rank blend) on the user's 20 training splits: 0.36 (0) / 0.37 (5) / 0.40 (10) / 0.44 (20) /
   0.52 (40) / 0.57 (80) vs ratings only 0.20 / 0.28 / 0.37 / 0.48 / 0.54; wins at n = 20 in 16/20
   -> by the rule it helps. This is the best images-only few-rating result so far.

### Crowd prior refinements (planned before running; `crowd_prior_eval.py`)
- Variants of the crowd face-beauty model by which SCUT faces it learns from: all; women
  (AF + CF); Caucasian women (CF); Asian women (AF). Chosen by zero-rating Spearman with the user's
  training ratings (training data only).
- Full-data CV (5 folds x 3): main model vs main + the chosen crowd score as an extra input
  (weighted x5). Adopted into the main model only if CV rises by 0.02 or more.
- Few-rating with the chosen variant: same 20 splits, herding picks, rank blend w = n / (n + k)
  for k = 10, 20, 40, 80; reported, k chosen by mean over n = 5, 10, 20, 40.

Results (job 12350256, `results/crowd_prior_eval.json`): all 5,500 faces 0.350 (strangers 0.336);
women 0.296; Caucasian women 0.312; Asian women 0.282 -> "all" chosen (more faces beat closer
faces). Full-data CV: main 0.595, main + crowd 0.596 -> not adopted (the full model already holds
it). Few-rating blend, mean over 20 splits: k=20 best: 0.37 (5) / 0.41 (10) / 0.46 (20) /
0.53 (40) / 0.57 (80) vs ratings only 0.20 / 0.28 / 0.37 / 0.48 / 0.54.

### Second crowd: MEBeauty (planned before running; `mebeauty_prior.py`)
MEBeauty (github fbplab/MEBeauty-database; research benchmark, no licence file, used only in this
private analysis): about 2,400 face crops of Black, Asian, Caucasian, Hispanic, Indian and Middle
Eastern women and men with average scores (1-10) from about 300 raters. Faces embedded with the
same OpenCLIP ViT-L-14. Priors compared by zero-rating Spearman with the user's training ratings:
SCUT, MEBeauty, and their average (standardized). The best is then used in the few-rating blend
(k = 20, herding picks, same 20 splits) and reported against the SCUT-only blend.

Results (job 12350571, `results/mebeauty_prior.json`): 2,365 MEBeauty faces. Zero-rating Spearman
with the user: SCUT 0.350 (strangers 0.336); MEBeauty 0.134 (0.188); average 0.262 (0.278).
SCUT alone stays the prior; MEBeauty is not used. (Its raters' consensus matches this user less.)

### Which people to rate first when a crowd prior exists (planned before running; `probe_prior.py`)
Model fixed: crowd prior + ratings, rank blend k = 20. Pick methods: random; kernel herding
(current); crowd-stratified (pool split into n equal bands of crowd score, the most typical person
of each band, i.e. nearest its band's centre in PCA-32); crowd extremes (n/2 highest and n/2 lowest
crowd scores). Tested on SCUT (60 raters x 5 splits, faces) and on the user's 20 training splits.
Rule (fixed now): a method replaces herding only if it beats herding at n = 20 for at least 45 of
60 SCUT raters and in at least 12 of the user's 20 splits.
Implementation note (03:20, before any result was printed): the first run rebuilt the same herding
kernel (3,850 faces, about 4 GB) for every rater and n on the shared login node and had done 2 of
5 SCUT splits after 40 minutes; it was stopped and restarted with the herding order computed once
per split (greedy, so its first n are the n-pick answer). Design and rule unchanged.

Results (login node, `results/probe_prior.json`), mean Spearman at n = 5 / 10 / 20 / 40:
SCUT: random 0.711 / 0.695 / 0.679 / 0.669; herding 0.712 / 0.699 / 0.689 / 0.677;
crowd-stratified 0.711 / 0.697 / 0.677 / 0.653; crowd extremes 0.713 / 0.706 / 0.692 / 0.676.
User: random 0.377 / 0.402 / 0.428 / 0.471; herding 0.371 / 0.414 / 0.459 / 0.528;
crowd-stratified 0.366 / 0.395 / 0.371 / 0.389; crowd extremes 0.359 / 0.386 / 0.429 / 0.384.
Wins over herding at n = 20: random SCUT 13/60, user 7/20; crowd-stratified 14/60, 4/20; crowd
extremes 37/60, 6/20 -> **herding stays**. Side finding: on SCUT the blend falls as ratings are
added (0.71 at 5 -> 0.68 at 40) because there the crowd prior alone (0.72) beats a rater's own
few ratings; k = 20 suits this user (prior 0.35), a rater who agrees with the crowd needs a
larger k. A per-rater k (e.g. from leave-one-out agreement between the prior and the ratings) is
an open idea, not tested.

### Per-person crowd weight (planned 03:32, before running; `adaptive_blend.py`)
Blend in standard units: score = (1 - w) z(prior) + w z(ratings model), z taken over the pool.
Adaptive w: for each w in 0, 0.1, ..., 1, leave-one-out over the n rated people (ridge refitted on
n - 1 at the full-n strength; prior fixed) -> Spearman of the blended LOO scores with the n
ratings; the best w is used (ties -> the larger w). Compared with: the current rank blend
k = 20, and the z-blend with fixed w = n / (n + 20). Herding picks, n = 10, 20, 40. SCUT: 60
raters x 5 splits (prior from the other 59 raters, pool faces only); user: the same 20 splits.
Rule (fixed now): adaptive replaces the fixed k = 20 if at n = 20 it beats it for at least 45 of
60 SCUT raters and its mean on the user's splits is no more than 0.01 below the fixed blend.

Results (login node, `results/adaptive_blend.json`), n = 10 / 20 / 40:
SCUT: rank k=20 0.699 / 0.689 / 0.677; z fixed 0.700 / 0.692 / 0.680; z adaptive 0.694 / 0.715 /
0.719. User: rank k=20 0.414 / 0.459 / 0.528; z fixed 0.411 / 0.455 / 0.524; z adaptive 0.365 /
0.404 / 0.484. Mean chosen w: SCUT 0.26 / 0.17 / 0.16; user 0.21 / 0.26 / 0.52 (fixed: 0.33 /
0.50 / 0.67). At n = 20 adaptive wins 52/60 SCUT raters but loses 0.056 on the user -> **keep
k = 20**. Leave-one-out with few ratings underrates the person's own model, so w comes out too
small for a rater who differs from the crowd.

### Half-way weight (planned 03:44, before running; `adaptive_blend.py --hybrid`)
w = average of the adaptive w and n / (n + 20); everything else as above. Same rule: replaces
k = 20 if at n = 20 it wins at least 45/60 SCUT raters and the user's mean is no more than 0.01
below the fixed blend. This is the last variant tried on this question tonight.

Results (`results/adaptive_blend_hybrid.json`), n = 10 / 20 / 40: SCUT 0.702 / 0.711 / 0.710;
user 0.395 / 0.439 / 0.511. At n = 20: wins 60/60 SCUT raters, user -0.021 -> **keep k = 20**
(by the rule). Open: a rater-level weight that does not cost this user; it needs more raters
like this user (who differ from the crowd) to tune, not more variants on the same data.

### Stock faces as the probe set (planned 2026-11-13 03:05, before the user rates any of them)
The user's idea of a fixed set of stock images that reveals a person's taste, tested with public
research faces instead of Instagram people. 40 SCUT-FBP5500 faces: kernel herding on PCA-32 of
their OpenCLIP L-14 codes, 30 from the women (AF + CF) and 10 from the men (AM + CM), seed-free
(herding is deterministic). `stock_probe.py pick` writes the list; the user rates them 0-10 on the
same rating page (optional, about 3 minutes). Then, with no Instagram ratings used for fitting:
(a) crowd prior alone (SCUT mean, all 5,500 faces; 0.350 on training accounts);
(b) ridge on the user's 40 stock ratings (leave-one-out strength) in the same face space, applied
    to the accounts' mean face crops;
(c) (a) and (b) averaged by rank, equal weights;
(d) ridge on the user's deviation from the crowd mean on those 40 faces, added to (a) (standardized).
Measure: Spearman with the user's ratings over all rated accounts (training + test; none were used
to fit), with a paired bootstrap 95% interval of the gain over (a). Rule (fixed now): stock faces
help if the best of (b)-(d) beats (a) by 0.05 or more and its interval excludes 0. If so,
onboarding becomes: 40 stock faces first, then pool probes. Faces only, so body taste is not
covered; that is a known limit.
Dry run with simulated ratings (crowd mean x 2 plus noise, then deleted): the script runs; on all
rated accounts the crowd prior scores 0.312, (b) 0.08, (d) 0.31 -> "no clear help", as expected
for ratings that only copy the crowd.

Results (user rated all 40 on 2026-11-13 morning, 2 undos, no skips; scores 4-7 only;
`results/stock_probe.json`): the user's stock ratings agree with the SCUT crowd mean on the same
40 faces at 0.895 (a SCUT rater vs the others: median 0.78). On all rated accounts: (a) crowd
prior 0.312; (b) stock ratings alone 0.112, gain CI [-0.28, -0.11]; (c) rank average 0.237;
(d) crowd + own deviation 0.312, gain +0.000 [-0.001, 0.001] -> **stock faces do not help**.
On studio faces the user's taste is the crowd's, so 40 ratings there add nothing.
Exploratory, training accounts, grouped CV (not a test, nothing adopted): a ridge on the
Instagram face crops alone 0.607; no-clothing body cut-outs alone 0.501; both (main) 0.595.
The user rates the whole person (2026-11-13: "face is just part of the overall rating"); the face
crops predict that overall rating because they carry cues that go with the whole person (a slim
face with a slim body, age, makeup, styling), not because the rating is about the face. The
body cut-outs are the noisier input (small, varied poses, partly hidden), which likely explains
their lower score. SCUT faces are neutral studio shots without that range, and the user used
only 4-7 on them. Probes that reveal this taste have to be whole-person photos like the pool's,
which is what the herding probes are.

### Onboarding page (2026-11-13, a tool, not a test)
`taste_onboard.py export` writes the Gram matrix of the standardized image codes, crowd scores,
herding order and opaque ids; the local page `taste_site/onboard.html` fits the same ridge in
kernel form (centred, strength by leave-one-out as RidgeCV) and the same k = 20 rank blend.
Checked against `taste_onboard.py predict` with the same 20 ratings: rank agreement 0.99999, max
score difference 0.002. Whole-pool check, run once with nothing tuned: the user's earlier ratings
of the first 20 herding people (whole pool) as input, Spearman of the ranking of the other 387
with the user's ratings 0.509 (crowd score alone 0.31). The training-split estimate was 0.46.
Live check (2026-11-13 morning): the user re-rated the same 20 people on the page. Fresh vs
earlier ratings of those 20: Spearman 0.906, 12 identical, 19 within 1 point, means 6.00 vs 5.85.
Ranking of the other 387 vs the user's earlier ratings: **0.424** (crowd alone 0.31). Rating
noise alone moves this a lot at n = 20: the earlier ratings with each one shifted by +-1 with
probability 0.4 (300 draws) give median 0.48, 5-95% 0.42-0.53; the live result sits at the 7th
percentile. Consistent with the 0.46 estimate, at its low end.

## Round 1 of improvements (user, 2026-11-13: "keep improving the model accuracy while requiring
## less ratings from me"). Planned before running; `improve_r1.py`; training accounts only.

Baselines, re-measured in the same script: full model = OpenCLIP L-14 + SigLIP (no-clothing
cut-out + face, account means), ridge, strength by the usual 5-fold x 3 grouped CV (0.636);
few-rating = SCUT crowd prior + herding picks + ridge on L-14 codes, rank blend k = 20, the same
20 pool/evaluation splits (0.371 / 0.414 / 0.459 / 0.528 / 0.567 at n = 5 / 10 / 20 / 40 / 80).

Full-data variants (rule: replaces the full model if CV is at least 0.02 higher):
- F1 ensemble: separate ridges on L-14, SigLIP and ViT-H (each no-clothing + face), predictions
  standardized and averaged.
- F2 photo-level: one row per photo (L-14 + SigLIP, no-clothing + face), labelled with the
  account's rating; an account's prediction is the mean over its photos.
- F3 robust account mean: drop the quarter of an account's photos (if it has 4+) least similar
  to the account's median L-14 no-clothing code, then average.
- F4 kernel ridge, RBF kernel (median-distance bandwidth x 0.5 / 1 / 2) on the standardized
  L-14 + SigLIP account codes.

Zero-rating prior (rule: adopted if zero-rating Spearman on training accounts rises by 0.03 or
more over SCUT alone, 0.350, and the k = 20 blend with it wins at n = 20 in 15+ of 20 splits):
- Z: zero-shot "attractiveness" from the image models' text towers, fixed prompts (not tuned):
  positive "a photo of a very attractive person", "a photo of a beautiful woman", "a photo of a
  handsome man", "a photo of a gorgeous model"; negative "a photo of an unattractive person",
  "a photo of a plain-looking woman", "a photo of a plain-looking man", "a photo of an
  ordinary-looking person". Photo score = mean similarity to positives minus negatives, on the
  no-clothing cut-out and the face; account mean; L-14 and SigLIP averaged (standardized).
  Unlike SCUT it sees the whole person. Combined prior = mean of standardized SCUT and Z.

Few-rating variants (rule: replaces the baseline if its mean over n = 5, 10, 20, 40 is 0.02 or
more higher and it wins at n = 20 in 15+ of 20 splits):
- R2 transfer ridge: the prior score as one extra input scaled x 30 (so ridge barely shrinks
  it) next to the L-14 codes; its prediction used directly, no blend.
- R3 low-dimensional ratings model: ridge on PCA-k of the pool's codes (unsupervised, all
  accounts), k from 4, 8, 16, 32, 64 by leave-one-out on the n ratings; rank blend k = 20.
- R4 switch: at n >= 40 the ratings model uses L-14 + SigLIP codes; below 40 as baseline.

Full-data results (job 12353321): baseline L-14 + SigLIP 0.636; F1 ensemble 0.631; F2 photo-level
0.652; F3 robust mean 0.621; F4 kernel ridge 0.632. Best F2, +0.016 -> below the 0.02 rule, the
full model stays. (F2 is the one to retest if more ratings arrive.) The job then stopped at the
zero-shot step: the SigLIP text tokenizer had never been downloaded and compute nodes are
offline. It was fetched on the login node and the prior and few-rating parts rerun alone (job
12353332, `--skip-full`, the full-data numbers above reused); nothing in the design changed.

Prior and few-rating results (job 12353332, `results/improve_r1.json`): zero-rating Spearman on
training accounts: SCUT 0.350; zero-shot prompts 0.046 (parts: L-14 cut-out 0.09, L-14 face 0.11,
SigLIP cut-out 0.03, SigLIP face -0.01); SCUT + zero-shot 0.233 -> **keep SCUT**. Generic
"attractive" wording in CLIP-type models says almost nothing about this rater; the SCUT
raters' scores carry the useful crowd signal.
Few-rating, n = 5 / 10 / 20 / 40 / 80: baseline 0.371 / 0.414 / 0.459 / 0.528 / 0.567; combined
prior 0.261 / 0.319 / 0.401 / 0.502 / 0.558; R2 transfer ridge 0.280 / 0.323 / 0.417 / 0.510 /
0.573; R3 PCA-k 0.348 / 0.374 / 0.449 / 0.493 / 0.553; R4 switch 0.371 / 0.414 / 0.459 / 0.537 /
0.586. By the rules none replaces the baseline (gains -0.072, -0.061, -0.027, +0.002). The R4
rule was badly set: R4 equals the baseline below n = 40 by design, so a rule on n = 5-40 and wins
at n = 20 could not detect it.

### Large-model switch at 40+ ratings (post-hoc follow-up, planned 08:55 after seeing the means)
Same 20 splits and picks; R4 vs baseline at n = 40, 80 and 120. Rule: the switch is adopted for
n >= 40 if it wins in at least 15 of 20 splits at both n = 40 and n = 80. Labelled post-hoc: the
means at 40 and 80 were seen before this rule was written.
Results (login node, `results/switch_check.json`): baseline 0.528 / 0.567 / 0.591 at n = 40 / 80
/ 120; switch 0.537 / 0.586 / 0.617; switch wins 10 / 15 / 17 of 20 -> not adopted (fails at
n = 40). From about 80 ratings the larger codes look better; that is a lead, not a decision.

## Free exploration (user, 2026-11-13 08:43: "keep free self-supervised free exploration till
## 10:10am"). Each test planned here before it runs; training accounts only.

### Identity space: self-supervised directions from same-person photos (`identity_space.py`)
No ratings are used to build it. Photo codes (L-14 no-clothing + face, 1,536 numbers) of all 412
people. Within-person scatter Sw (each photo minus its person's mean) and between-person scatter
Sb (person means); Sw regularised with 1% of its mean eigenvalue. Directions = generalised
eigenvectors of (Sb, Sw): those along which people differ more than one person's photos do.
Kept: every direction with ratio > 1 (rule fixed now, no ratings involved); people are the mean
of their photos projected onto them, standardized. Also reported, not used for the decision:
the top 32 / 64 / 128 / 256 directions.
- Few-rating: same 20 splits, the same herding picks as the baseline (picked on the original
  codes), ridge on identity-space codes, rank blend with the SCUT prior at k = 20.
  Rule: replaces the ratings model if its mean over n = 5, 10, 20, 40 is 0.02 or more higher
  and it wins at n = 20 in 15+ of 20 splits.
- Full data: 5-fold x 3 grouped CV, ridge; rule: +0.02 over the same codes without the
  projection (L-14 0.595).
Results (login node, `results/identity_space.json`): 130 of 1,536 directions have ratio > 1
(top 38, 15, 12). Full-data CV: L-14 0.595; identity (130) 0.493; top 32 0.530; 64 0.511; 128
0.500; 256 0.366. Few-rating at n = 5 / 10 / 20 / 40 / 80: baseline 0.371 / 0.414 / 0.459 /
0.528 / 0.567; identity (130) 0.366 / 0.378 / 0.376 / 0.395 / 0.432; top 32 0.378 / 0.414 /
0.456 / 0.489 / 0.515. -> **not adopted** (few gain -0.064, wins 4/20; full -0.101). The
directions that separate one person from another across their photos are mostly facial-identity
detail, not what the rater's taste follows; standardizing them gives that detail as much weight
as the broad traits.

### Who to rate next (`active_next.py`)
After the 20 onboarding people, which 20 / 40 more? Same 20 splits; base = the first 20 herding
picks of the pool. Strategies, adding 5 people at a time (refitting between steps where the
strategy uses ratings): herding continuation (current); random; uncertainty (largest ridge
posterior variance given who is rated, kernel form, strength from the current fit; no ratings
used); disagreement (largest gap between the crowd score's rank and the ratings model's rank
among the unrated pool); top-predicted (highest current blended score; rates the people most
likely to be liked). Model and measure as before (k = 20 rank blend, Spearman on the evaluation
30%). Rule: a strategy replaces herding continuation if it wins at both 40 and 60 ratings in
15+ of 20 splits.
Results (login node, `results/active_next.json`), Spearman at 40 / 60 ratings: herding 0.528 /
0.557; random 0.487 / 0.511; uncertainty 0.485 / 0.502; disagreement 0.496 / 0.529;
top-predicted 0.492 / 0.533. Wins over herding at most 10/20 -> **herding continuation stays**.
Picking people to cover the pool beats picking by the model's doubts at these sizes.

### Few-rating model trained on single photos (`photo_fewshot.py`)
Round 1's best full-data variant (one row per photo, +0.016) applied to few ratings: the ratings
model is a ridge on the rated people's photos (L-14 no-clothing + face per photo, each labelled
with its person's rating, strength by leave-one-person-out), a person's prediction = the mean of
their photos' predictions; same picks, prior and k = 20 blend. Rule: replaces the ratings model
if its mean over n = 5, 10, 20, 40 is 0.02 or more higher and it wins at n = 20 in 15+ of 20.
Results (login node, `results/photo_fewshot.json`): photo-level 0.368 / 0.413 / 0.455 / 0.526 /
0.584 vs baseline 0.371 / 0.414 / 0.459 / 0.528 / 0.567 -> **not adopted** (gain -0.002, wins
9/20). Like the larger codes, it only pays from about 80 ratings.

### Typicality as a free prior (`prior_typical.py`)
Faces close to the population average tend to be rated as more attractive (an old finding in
face research). No ratings needed: a person's typicality = cosine similarity of their mean
L-14 face code to the mean face code of all 412 people; the same for the no-clothing cut-out;
and the two averaged (standardized). Measured by zero-rating Spearman on training accounts.
Rule: the best of the three is added to the SCUT prior (mean of standardized scores) only if
that raises zero-rating Spearman by 0.03 or more over SCUT alone (0.350) and the k = 20 blend
with it wins at n = 20 in 15+ of 20 splits.
Results (login node, `results/prior_typical.json`): zero-rating Spearman, training accounts:
face typicality 0.214, body 0.159, both 0.195 (SCUT 0.350; the two correlate 0.23); SCUT + face
typicality 0.352 -> **not adopted** (gain +0.002; blend wins 10/20). The averageness effect is
real for this rater (0.21 with no ratings and no outside data) but SCUT already holds it.

### SCUT raters as stand-in members (`prior_raters.py`)
Instead of the SCUT mean, one ridge model per SCUT rater (60, on all 5,500 faces, L-14 codes),
each applied to the pool's mean face codes. With n >= 5 ratings, rater r gets weight
max(0, rho_r)^2, rho_r = Spearman of its scores with the user's n ratings; prior = weighted mean
of the raters' standardized scores (the plain mean when all weights are 0). Then the usual k = 20
blend with the ratings model. Same 20 splits and picks. Rule: replaces the mean prior if its
mean over n = 5, 10, 20, 40 is 0.02 or more higher and it wins at n = 20 in 15+ of 20 splits.
Results (login node, `results/prior_raters.json`): single SCUT raters vs the user with zero
ratings: median 0.303, best 0.455 (chosen on all training ratings, so optimistic), worst 0.058;
mean prior 0.349. Weighted prior 0.369 / 0.421 / 0.464 / 0.531 / 0.568 vs mean 0.371 / 0.415 /
0.459 / 0.528 / 0.567 -> gain +0.003, wins 17/20 -> **not adopted** (consistent but tiny).

### Sharper rater weights (post-hoc follow-up, labelled; same script, `--sharp`)
Planned after seeing the weighted-prior result and the best single rater (0.455). Two sharper
weightings, both from the n ratings only: top-5 (the mean of the 5 raters with the highest rho_r)
and softmax (weights exp(20 rho_r)). Same rule as above; post-hoc, so a pass is a lead to
confirm with new raters, not a change to the model.
Results (`results/prior_raters_sharp.json`): top-5 0.347 / 0.416 / 0.461 / 0.532 / 0.570;
softmax 0.332 / 0.413 / 0.462 / 0.533 / 0.568 (gains -0.004, -0.008) -> no. With 5-20 ratings the
rater most like this user cannot be picked out reliably; the 0.455 rater stays out of reach.

### Face / body weighting in the few-rating model (`block_weight.py`)
The ratings model's input is [no-clothing cut-out, face] (standardized). Variants scale the two
blocks before the ridge: face only; face x2; equal (baseline); body x2; body only. The rating
stays the whole person; only the model's weighting of its inputs changes. Same picks, prior,
k = 20 blend and 20 splits. Rule: the best variant replaces equal weighting if its mean over
n = 5, 10, 20, 40 is 0.02 or more higher and it wins at n = 20 in 15+ of 20 splits.
Results (`results/block_weight.json`), n = 5 / 10 / 20 / 40 / 80: face only 0.367 / 0.416 /
0.459 / 0.519 / 0.574; face x2 0.369 / 0.417 / 0.464 / 0.524 / 0.574; equal 0.371 / 0.414 / 0.459
/ 0.528 / 0.567; body x2 0.370 / 0.403 / 0.439 / 0.498 / 0.542; body only 0.367 / 0.391 / 0.413 /
0.465 / 0.491 -> **equal stays** (best face x2, gain +0.000). The cut-out codes are the noisier
input (pose, framing); they add to the face codes but should not outweigh them.

### Mirror averaging (`flip_tta.py`, GPU)
Each no-clothing cut-out and face crop is also embedded mirrored (L-14); a photo's code = the
normalised mean of the two. Nothing else changes. Rule: adopted if full-data CV rises by 0.02 or
more over L-14 (0.595), or the few-rating curve passes the usual rule (mean over n = 5-40 +0.02
and 15+ of 20 wins at n = 20).

### Coarser answers: like / pass, or low / mid / high (`coarse_ratings.py`)
Less effort per rating instead of fewer ratings. The probes' 0-10 ratings are coarsened before
fitting: binary (above the rater's median of the n picked ratings = 1, else 0; ties at the
median go to 0) and three levels (terciles of the n picked ratings). Model, picks, prior and
blend unchanged; always scored against the full 0-10 ratings of the evaluation 30%.
Rule (fixed now): a coarse format is "good enough" if its mean over n = 5, 10, 20, 40 is within
0.02 of the 0-10 baseline; also reported, the n at which binary matches 0-10 at n = 20.
Results (`results/coarse_ratings.json`), n = 5 / 10 / 20 / 40 / 80 / 120: 0-10 0.371 / 0.414 /
0.459 / 0.528 / 0.567 / 0.591; low / mid / high 0.369 / 0.398 / 0.445 / 0.485 / 0.548 / 0.575;
like / pass 0.370 / 0.411 / 0.425 / 0.442 / 0.480 / 0.520. Three levels: gap 0.019 -> "good
enough" by the rule, but it falls behind from 40 on. Like / pass: gap 0.031, and it needs 80
answers to match 20 ratings on 0-10 -> **keep 0-10**. Simulated by coarsening real 0-10 ratings;
real yes / no answers may differ.

Mirror averaging results (job 12353387, `results/flip_tta.json`): a photo and its mirror agree at
cosine 0.977 (cut-out) / 0.970 (face); full CV 0.595 -> 0.595; few-rating identical (wins 8/20)
-> **not adopted**.

### Where the full model goes wrong (`residuals.py`, exploratory, no rule)
CV predictions of L-14 + SigLIP (ridge 3000; Spearman 0.647). Rank error vs properties: owner
photos -0.07, woman (zero-shot) +0.03, photo consistency +0.02, face consistency +0.03, LAION
photo aesthetic -0.01, rating extremeness -0.16. Model Spearman by halves: mid-rated people 0.47
vs extreme-rated 0.78; more polished photos (LAION aesthetic above median) 0.57 vs 0.70; women
0.64 (n = 288), men 0.72 (n = 17). No data-quality fault to fix; the hard cases are people the
rater put in the middle and people with polished photos.

### Aligning the crowd model to Instagram faces (`prior_domain.py`)
SCUT faces (studio) and the pool's face crops (Instagram) sit in different parts of the code
space; the crowd ridge was fitted on SCUT-standardized codes and applied to pool codes with the
same scaling. Unsupervised fixes (no ratings): (a) centre each domain on its own mean (pool
mean from all 3,526 face crops), SCUT scaling; (b) standardize each domain separately; (c) CORAL:
whiten the pool face codes with their own covariance (+ mean eigenvalue x 0.1 on the diagonal)
and recolour with SCUT's. Also (d) kernel ridge (RBF, median bandwidth, strength by 5-fold CV on
SCUT) instead of linear ridge. Rule: the best is adopted if zero-rating Spearman on training
accounts rises by 0.03 or more over 0.350 and the k = 20 blend with it wins at n = 20 in 15+ of
20 splits.
Results (login node, `results/prior_domain.json`): zero-rating Spearman: as now 0.350; (a)
0.350; (b) 0.358; (c) CORAL 0.325; (d) kernel ridge 0.358. Blend with (d): 0.384 / 0.426 /
0.476 / 0.537 / 0.571 vs 0.371 / 0.414 / 0.459 / 0.528 / 0.567 (wins 16/20 at n = 20) ->
**not adopted** by the rule (zero-rating gain +0.008 < 0.03), though the blend gain is
consistent (mean +0.013 over n = 5-40).

### Kernel-ridge crowd model on per-domain standardized codes (post-hoc follow-up, labelled)
Planned 09:06 after seeing (b) and (d): kernel ridge as in (d), fitted on SCUT codes standardized
by SCUT's own statistics and applied to pool face codes standardized by the pool's (as in (b)).
Judged on the few-rating curve, as the prior's purpose is the blend: adopted if its mean over
n = 5, 10, 20, 40 is 0.02 or more above the current prior and it wins at n = 20 in 15+ of 20.
Post-hoc: a pass is a lead to confirm (new raters or new accounts), not yet a model change.
Results (`results/prior_domain_combo.json`): zero-rating 0.382 (vs 0.350); blend 0.414 / 0.450 /
0.479 / 0.540 / 0.573 vs 0.371 / 0.414 / 0.459 / 0.528 / 0.567; mean gain +0.028, wins 14/20 ->
misses the rule by one split.

### Confirming the aligned crowd model on held-out accounts (planned 09:09)
The crowd model uses no ratings of this rater, and the 99 test accounts played no part in
choosing it, so their ratings can check it cleanly (this adds a look at the test set for the
prior only; the main model's clean result stays 0.668). Measure: zero-rating Spearman on the
99 test accounts, current crowd model vs the aligned kernel-ridge one, with a paired bootstrap
interval of the difference. Rule (fixed now): confirmed, and adopted as the crowd model, if the
gain is 0.03 or more and its 95% interval excludes 0.
Results (`results/prior_confirm.json`): test accounts (n = 99): current 0.189, aligned kernel
ridge 0.212, gain +0.023 [-0.062, 0.113] -> **not confirmed; the current crowd model stays**.
The crowd model is weaker on the test accounts than on the training ones (0.19 vs 0.35), a
reminder of how noisy a zero-rating score on about 100 people is.

### Semi-supervised: ratings spread along the pool's neighbour graph (`laprls.py`, planned 09:10)
Laplacian-regularised least squares (Belkin, Niyogi and Sindhwani 2006) with a linear kernel on
the standardized L-14 codes of all 412 people (unrated people included, no ratings of theirs
used). Graph: 10 nearest neighbours in PCA-32, symmetrised, binary weights, unnormalised
Laplacian. Coefficients c = (J K + a I + g L K)^-1 J y over all people (J marks the rated);
a from 13 values (1e-3 .. 1e3), g from {0, 0.01, 0.1, 1} (g = 0 is plain kernel ridge), both
chosen by leave-one-out on the n ratings (hat matrix). Predictions blended with the SCUT prior,
k = 20, same picks and 20 splits. Rule: replaces the ratings model if its mean over n = 5, 10,
20, 40 is 0.02 or more higher and it wins at n = 20 in 15+ of 20 splits.
Results (login node, `results/laprls.json`): LapRLS 0.370 / 0.416 / 0.464 / 0.521 / 0.561 vs
baseline 0.371 / 0.414 / 0.459 / 0.528 / 0.567 (graph weight chosen > 0 in 25-65% of fits) ->
gain -0.000, wins 11/20 -> **not adopted**. The pool's neighbour structure adds nothing the
ridge on the codes does not already use.

### Does the alignment help on an independent face set? (`prior_mebeauty_check.py`, 09:11)
MEBeauty (2,365 faces, other raters, more ethnicities; codes already in `datasets/mebeauty/me.npz`)
as a third domain with its own crowd scores. The SCUT crowd model, current form vs aligned
kernel ridge (MEBeauty codes standardized by MEBeauty's own statistics), scored against
MEBeauty's mean ratings. Rule: the alignment idea is supported if the gain is 0.03 or more with
a 95% bootstrap interval excluding 0. This is evidence about the method, not about this rater;
the crowd model stays as it is unless the user decides otherwise.
Results (`results/prior_mebeauty_check.json`, 2,365 faces): Spearman with MEBeauty means:
current 0.611; per-domain standardization only 0.618; kernel ridge only 0.572; aligned kernel
ridge 0.586; gain -0.025 [-0.034, -0.016] -> **not supported**. Kernel ridge transfers worse
across face sets; its gain on the user's training accounts was most likely selection luck
among several variants. The lead is closed; the linear crowd model stays.

### Summary of the free exploration (08:43-09:15)
Twelve pre-registered tests (three of them labelled follow-ups), none adopted: identity space,
who to rate next, typicality prior, weighted SCUT raters (and sharper weights), face / body
weighting, single-photo few-rating model, mirror averaging, coarse answers, crowd-model
alignment (training, held-out and MEBeauty checks), and semi-supervised graph spreading. With
5-40 ratings the current method (crowd prior + herding + ridge on L-14 codes, k = 20) was not
beaten. Findings worth keeping: face averageness alone gives 0.21 with no data at all; the SCUT
rater closest to this user reaches 0.455 but cannot be found from 20 ratings; three-level answers
are nearly as good as 0-10 up to 20 ratings, like / pass is not; past about 80 ratings, larger
codes and single-photo training both help (0.58-0.59 vs 0.57 at 80). The model's hardest cases
are people the rater placed mid-scale and people with polished photos.

### Full model, round 2 (post-hoc follow-up to round 1's F2, planned 09:12; `improve_r2.py`)
Round 1's photo-level model (L-14 + SigLIP, 0.652) came closest. Two variants, same 5-fold x 3
grouped CV, ridge strength from the usual 10 values (x10 for photo rows): G1 photo-level with
L-14 + SigLIP + ViT-H (no-clothing + face); G2 the mean of the standardized predictions of the
account-level L-14 + SigLIP model and the photo-level L-14 + SigLIP model. Rule: replaces the
full model if CV reaches 0.656 (baseline 0.636 + 0.02). Labelled post-hoc; a pass is to be
confirmed with new ratings before the onboarding page or the report changes.
Results (login node, `results/improve_r2.json`): G1 0.654; G2 0.653 -> **neither reaches 0.656;
the full model stays**. Photo-level training is worth about +0.017 (0.652-0.654 in three
variants), consistently but below the rule.

### Shared taste dimensions from other raters (SCUT simulation; `multi_rater_scut.py`, 09:15)
What other members' ratings could add. SCUT, 60 raters, the same 5 image splits (70% pool,
30% evaluation) and herding picks as before. For each rater r: ridge weight vectors of the other
59 raters on the pool images (one shared strength, the crowd model's); taste basis = their mean
plus the top k principal directions of their weights. The new rater's model: a small ridge (LOO
strength) on the k + 1 basis scores of the n rated images, k chosen by leave-one-out among
0, 2, 5, 10. Compared with the current method (crowd prior + ridge on all codes, rank blend
k = 20) at n = 5, 10, 20, 40. Rule (fixed now): the shared-dimension model is better if it wins
at n = 20 for 45+ of 60 raters. A pass says "members' ratings would help a new rater"; it
changes nothing for the current single-rater model.
Results (login node, `results/multi_rater_scut.json`): current 0.712 / 0.699 / 0.689 / 0.677 at
n = 5 / 10 / 20 / 40; shared dimensions 0.610 / 0.653 / 0.689 / 0.706 (k = 0 chosen in 51% of
fits); wins at n = 20: 33/60 -> **not better** by the rule. Worse with 5-10 ratings (it re-fits
even the crowd part from a handful of ratings), better at 40.

### Anchored shared dimensions (post-hoc follow-up, labelled; `--anchored`, planned 09:18)
Fix for the small-n weakness: the crowd score enters as a fixed offset (standardized, slope 1)
and only deviations along the top k other-rater directions are fitted (ridge, LOO strength,
k by LOO from 2, 5, 10), on standardized ratings; with no signal it falls back to the crowd.
Same rule (45+ of 60 at n = 20). Post-hoc: a pass is evidence for collecting members'
ratings, not a model change.
Results (`results/multi_rater_scut_anchored.json`): anchored 0.656 / 0.700 / 0.706 / 0.718 vs
current 0.712 / 0.699 / 0.689 / 0.677 at n = 5 / 10 / 20 / 40; wins at n = 20: 46/60 -> passes
(post-hoc). Weaker at 5 ratings, better from 20, and it keeps improving where the current blend
declines. Reading: a rater's departure from the crowd lies along a few directions that other
raters share, so other raters' ratings make a new rater's personal part learnable from about
20 ratings (it was not learnable from the rater's own ratings alone; see "Many raters").
[Corrected 13:20, see "Afternoon exploration", D: this gain came from anchoring on the crowd,
not from other raters' directions; anchoring alone does as well on SCUT.]

### Do SCUT's shared directions carry over to this rater? (`--user`, planned 09:22)
The anchored model on the user's 20 training splits: anchor = SCUT crowd score of the pool's mean
face codes; deviations fitted along the top k directions of the 60 SCUT raters' face-taste
weights (k from 2, 5, 10 by LOO), on standardized ratings; same herding picks. Compared with the
current method. Rule: carries over if its mean over n = 5, 10, 20, 40 is 0.02 or more higher and
it wins at n = 20 in 15+ of 20 splits.
Results (`results/multi_rater_user.json`): anchored SCUT directions 0.255 / 0.363 / 0.378 /
0.408 / 0.443 vs current 0.371 / 0.414 / 0.459 / 0.528 / 0.567 -> gain -0.092, wins 5/20 ->
**do not carry over**. The useful shared directions have to come from raters of the same kind of
photos (whole people, social-media pool), not from studio faces.

### Where the exploration leaves the few-rating goal (09:25)
With one rater, the current method (crowd prior + herding + ridge on L-14 codes, k = 20) was not
beaten at 5-40 ratings by any test this morning or overnight. The one approach that made a rater's personal taste
learnable from about 20 ratings was anchoring on the crowd and fitting deviations along
directions shared by other raters (SCUT: 46/60 raters), and those directions must come from
raters of the same photos. Next step that can lower the ratings a new person needs: several
people (members) rate the same pool, 40-80 each, then the anchored shared-direction model is
tested with each member left out in turn.
[Corrected 13:20: the SCUT evidence for this step does not hold (see "Afternoon exploration",
D). Whether members' ratings help a rater who differs from the crowd is an open question.]

## Free exploration, second window (user, 2026-11-13 09:30: "keep exploring until 10:10am")

### Crowd-anchored residual model (`anchored_residual.py`, planned 09:32)
Target: the rater's standardized ratings minus the standardized crowd score of the same people;
ridge on the L-14 codes (LOO strength) fitted to that difference from the n ratings; prediction
= crowd score + predicted difference (with no signal it falls back to the crowd). Compared with
the current rank blend (k = 20) on the user's 20 splits, and on SCUT (60 raters x 5 splits,
crowd = the other raters' mean) for reference. Rule: replaces the rank blend if, on the user's
splits, its mean over n = 5, 10, 20, 40 is 0.02 or more higher and it wins at n = 20 in 15+ of 20.

### Face-recognition codes as an extra input (`sface_input.py`, planned 09:32)
The owner step stored SFace embeddings (128 numbers per owner face, a face-recognition model,
unlike CLIP's general-purpose codes). Account mean, normalised, standardized, appended to the
codes. Full data: L-14 + SigLIP + SFace vs L-14 + SigLIP (0.636), rule +0.02. Few ratings:
L-14 + SFace vs L-14 in the ratings model, usual rule (mean over n = 5-40 +0.02, 15+ of 20 at
n = 20). SFace alone reported.
Results (login node): anchored residual, user 0.341 / 0.376 / 0.431 / 0.474 / 0.527 vs blend
0.371 / 0.414 / 0.459 / 0.528 / 0.567 (wins 8/20) -> **keep the blend**; on SCUT it wins for
58/60 raters (0.714 vs 0.689 at n = 20). Same split as the per-person weight: anchoring suits
raters close to the crowd, blending suits this rater. SFace: alone 0.566 (full CV); L-14 +
SigLIP + SFace 0.645 vs 0.636 (+0.010); few-rating L-14 + SFace 0.372 / 0.418 / 0.466 / 0.539 /
0.583 vs 0.371 / 0.414 / 0.459 / 0.528 / 0.567 (+0.006, wins 17/20) -> **not adopted**, small
but consistent.

### Crowd model on face-recognition codes (`sface_crowd.py`, planned 09:36)
Face-recognition codes are trained to ignore lighting, make-up and photo style, so a crowd model
on them may carry over from SCUT's studio faces to social-media faces better than one on CLIP
codes. SCUT faces embedded with YuNet + SFace (as in the owner step); ridge (LOO strength) on
the standardized SFace codes to SCUT's mean scores; applied to each pool account's mean owner
SFace code. Variants: SFace crowd alone; mean of standardized SFace and current crowd scores.
Rule: the better variant is adopted if zero-rating Spearman on training accounts rises by 0.03 or
more over 0.350, the k = 20 blend with it wins at n = 20 in 15+ of 20 splits, and its zero-rating
gain on the 99 test accounts is positive.
Results (login node, `results/sface_crowd.json`; SCUT faces embedded on CPU, about 90 minutes,
5,500 / 5,500 with a code): zero-rating Spearman, training / test accounts: current (CLIP)
0.350 / 0.189; SFace crowd alone 0.309 / 0.343; SFace + current 0.386 / 0.298. Blend with SFace
+ current: 0.414 / 0.455 / 0.483 / 0.547 / 0.575 vs 0.371 / 0.414 / 0.459 / 0.528 / 0.567 at
n = 5 / 10 / 20 / 40 / 80; wins at n = 20: 18/20. Training gain +0.035, test gain +0.110 ->
**adopted**: the crowd prior becomes the mean of the standardized CLIP and SFace crowd scores.
Face-recognition codes carry the crowd's taste from studio faces to social-media faces better
than CLIP codes do (held-out accounts 0.343 vs 0.189), and the two crowd scores add up.
Onboarding page re-checked with the new crowd score (export re-run; whole pool, labelled: the
held-out accounts took part in the adoption rule): crowd score alone vs the rater's ratings, all
rated accounts 0.364; the user's earlier ratings of the 20 herding people as input -> 0.548 on
the other 387 (was 0.509); the user's fresh re-ratings of the same 20 from this morning -> 0.458
(was 0.424).

## Onboarding at 40 ratings, and pushing that number down (user, 2026-11-13)

User: "change onboarding to 40 ratings, also check if we can push that number down". The
onboarding page and `taste_onboard.py probes` now ask for 40 people (the first 20 unchanged;
earlier saved ratings carry over). Whole-pool check, single run, labelled: the user's earlier
ratings of the 40 herding people -> 0.520 on the other 368 (0.548 from the first 20 on 387; one
ordering, so noisy; the split means are the reliable figures).

### Pushing it down (`push_down.py`, planned 11:20, GPU for FaceNet)
Baseline: the adopted method (CLIP + SFace crowd prior, herding, ridge on L-14 codes, rank blend
k = 20): 0.414 / 0.455 / 0.483 / 0.547 / 0.575 at n = 5 / 10 / 20 / 40 / 80. Target: fewer
ratings to reach 0.547. Reported for every variant: n*, the number of ratings at which its mean
curve reaches 0.547 (linear in log n between measured points).
- T1 blend weight retuned for the new prior: k = 10, 40, 80 (k = 20 was chosen with the old one).
- T2 ratings model on L-14 + SFace account codes (SFace was a small, consistent help as input).
- T3 third face code in the crowd prior: FaceNet (Inception-ResNet-V1, VGGFace2 weights, in
  `idscore/facenet.py`) embeds SCUT's face-centred images and the pool's face crops as they are
  (no alignment); a crowd ridge on SCUT FaceNet codes; prior = mean of the three standardized
  crowd scores.
Rules (fixed now): T1 and T2 are adopted if their mean over n = 5, 10, 20, 40 is 0.02 or more
above the baseline and they win at n = 20 in 15+ of 20 splits. T3 is adopted if the prior's
zero-rating Spearman on training accounts rises by 0.03 or more, the blend wins at n = 20 in 15+
of 20 splits, and the zero-rating gain on the 99 test accounts is positive. Adopted pieces are
then combined and the combination's n* reported.
Results (job 12354531, `results/push_down.json`): zero-rating, training / test: CLIP + SFace
0.386 / 0.298; FaceNet alone 0.107 / 0.071; three codes 0.298 / 0.251. Curves (n = 5 / 10 / 20 /
40 / 80) and n*: baseline 0.414 / 0.455 / 0.483 / 0.547 / 0.575, n* 40; k = 10: n* 60; k = 40:
0.408 / 0.436 / 0.475 / 0.551 / 0.589, n* 38; k = 80: n* 56; T2 L-14 + SFace input 0.415 / 0.458
/ 0.492 / 0.559 / 0.594, n* 35 (gain +0.006, wins 18/20); T3 0.320 / 0.371 / 0.440 / 0.522 /
0.567, n* 59. -> **none adopted**: k = 20 stays; T2 is consistent (17/20 and 18/20 in two runs)
but under the 0.02 rule; FaceNet on unaligned crops carries almost no crowd taste (0.11), so T3
fails. Nothing combined.

Live check at 40 (user, 2026-11-13 ~11:40): the user rated people 21-40 on the page (1-20 from
the morning's re-rating). Fresh vs earlier ratings of the 40: Spearman 0.854 (the new 20:
0.763), 38 of 40 within one point. Ranking of the other 368 vs the user's earlier ratings:
**0.544** (training-split estimate at 40: 0.547).
User decision (2026-11-13): keep the rule; the SFace ratings-model input (about 35 ratings
instead of 40, under the 0.02 bar) is not adopted. Onboarding stays at 40 ratings.

## Afternoon exploration (user, 2026-11-13 ~11:50: "keep exploring until 3pm")

### A. Full model: single photos plus face-recognition codes (`afternoon_ab.py`, planned 13:00)
Single-photo training (+0.016-0.018) and SFace codes (+0.010) each helped the full model a
little. A: photo-level ridge on L-14 + SigLIP photo codes with each photo's account-mean SFace
code appended (standardized as one block), account prediction = mean over its photos; same
5-fold x 3 grouped CV and strength grid (x10) as round 1. Rule: replaces the full model if CV
reaches 0.656 (0.636 + 0.02).

### B. Onboarding picks that also cover face-recognition space (same script)
Herding on PCA-32 of [L-14 codes, SFace codes] (standardized, SFace block weighted to equal
total variance) instead of L-14 alone; adopted prior, ratings model and k = 20 unchanged; same
20 splits. Rule: replaces the picks if the mean over n = 5, 10, 20, 40 is 0.02 or more higher
and it wins at n = 20 in 15+ of 20 splits; n* (ratings to reach 0.547) reported.

### C. FaceNet on aligned faces as a third crowd code (`facenet_aligned.py`, planned 13:08, GPU)
The unaligned FaceNet test failed (0.11). Here both domains are aligned the same way from eye
positions, no detection: SCUT pupils are landmarks 58 / 59 of its 86-point files; pool photos use
the owner's COCO eye keypoints from `owner.json` on the original photo. Crop: eyes level, side
3.0 x the eye distance, centred 0.35 eye distances below the eye midpoint, 160 x 160. FaceNet
(VGGFace2) codes; crowd ridge on SCUT; pool account means. Rule as for T3: adopted into the prior
(mean of three standardized crowd scores) if zero-rating Spearman on training accounts rises by
0.03 or more over CLIP + SFace (0.386), the k = 20 blend wins at n = 20 in 15+ of 20 splits, and
the zero-rating gain on the 99 test accounts is positive. (Run note: one SCUT landmark file,
CM152, is empty; that image uses SCUT's mean pupil positions. Fixed after a crash, before any
result.)

### D. How many members, rating how many people? (SCUT simulation; `members_sim.py`, 13:15)
Realistic version of "Shared taste dimensions": SCUT, 5 image splits as before. For each new
rater r: M members drawn at random from the other 59 raters (M = 5, 10, 20), each "rating" only
the first R herding images of the pool (R = 40, 80, 160, the same images for every member);
each member's ridge weights from those R ratings; shared directions = top principal directions
of the members' weights around their mean. Crowd anchor = the other raters' mean (as before).
New rater: n = 10, 20, 40 ratings on the first n herding images; anchored model (crowd + fitted
deviation along the top k member directions, k from 1, 2, 5 by leave-one-out) vs the current
blend. Planning rule (fixed now): the smallest M x R at which the anchored model beats the
current blend at n = 20 for 45+ of 60 raters is what to ask members for; if none, members'
ratings at these sizes would not help a new rater.

Results, afternoon (A / B on the login node, C job 12356250, D on the login node):
- **A adopted.** Photo-level L-14 + SigLIP + SFace: CV 0.669 (strength 10,000) >= 0.656 -> the
  full model (all ratings) becomes this one (`results/afternoon_ab.json`).
- B: L-14 + SFace picks 0.407 / 0.433 / 0.489 / 0.513 / 0.537 vs 0.414 / 0.455 / 0.483 / 0.547 /
  0.575 (gain -0.014, wins 11/20) -> keep the L-14 picks.
- C: FaceNet on aligned faces: alone 0.039 (training) / 0.162 (test); three-code prior 0.318 /
  0.350 vs 0.386 / 0.298; blend wins 7/20 -> **no**. Of the face-recognition models only SFace
  carries the crowd's taste over (`results/facenet_aligned.json`).
- D (`results/members_sim.json`): every M x R passes the planning rule (51-55 of 60 raters at
  n = 20; smallest 5 members x 40 ratings: 0.711 / 0.717 / 0.718 at n = 10 / 20 / 40 vs the
  current blend 0.699 / 0.689 / 0.677). **But the comparison was the wrong one.** The members
  model is no better than the SCUT crowd alone (about 0.72) or than crowd anchoring with no
  members at all (`anchored_residual.py`: 0.714 at n = 20). On SCUT, whose raters agree with
  their crowd, anything that leans on the crowd beats the blend; the members' directions add
  nothing measurable. **Correction**: the same holds for "Shared taste dimensions" above (46/60,
  0.706 at n = 20): that gain came from anchoring on the crowd, not from other raters'
  directions. SCUT cannot show whether members would help this rater, who differs from the
  crowd far more than SCUT's raters do; that stays an open question that only real members'
  ratings can answer.

### Labelled look at the test set for the adopted full model (planned 13:15)
The held-out test set has been used before (clean result 0.668 for the original model, then a
labelled second look). This third look fits the adopted full model on all training accounts
(strength 10,000) and scores the 99 test accounts once, with a bootstrap interval. Nothing is
decided by it; it is reported so the full model's number is not only a CV figure.
Result (`results/full_test_look.json`): **0.723 [0.603, 0.811]** on the 99 test accounts
(labelled third look; clean result 0.668 for the original model, second look 0.716 for L-14 +
SigLIP). `taste_full.py` packages the adopted full model: it scores every person (rated people
out-of-fold); out-of-fold agreement on all 407 rated people 0.700 (random folds, so the seven
same-person pairs may share a fold).

### E. A second crowd through face-recognition codes: MEBeauty on SFace (`mebeauty_sface.py`, 13:27)
MEBeauty failed as a crowd on CLIP codes (0.134). SFace codes carried SCUT's crowd over far
better than CLIP codes, so MEBeauty's 2,365 faces are embedded with YuNet + SFace (as for
SCUT; split over 4 CPU processes), a crowd ridge is fitted to MEBeauty's mean scores, applied to
the pool's SFace codes, and added to the prior (mean of standardized CLIP-SCUT, SFace-SCUT and
SFace-MEBeauty scores). Rule (as for the SFace crowd): adopted if zero-rating Spearman on
training accounts rises by 0.03 or more over 0.386, the k = 20 blend wins at n = 20 in 15+ of 20
splits, and the zero-rating gain on the 99 test accounts is positive.
Result (`results/mebeauty_sface.json`; all 2,365 faces got an SFace code): MEBeauty on SFace
alone 0.135 (training) / 0.014 (test); three-score prior 0.317 / 0.200 vs 0.386 / 0.298; blend
0.341 / 0.390 / 0.446 / 0.525 / 0.565 at n = 5 / 10 / 20 / 40 / 80, wins 0/20 -> **no**. SFace
codes did not rescue MEBeauty: its crowd (mixed ages and ethnicities, ordinary photos) does not
match this rater the way SCUT's does, whichever code reads the face.

### Afternoon summary
One change adopted: the full model (A, CV 0.669, labelled test look 0.723). Nothing new helps
the few-rating side; three ways of adding more crowd (FaceNet unaligned and aligned, MEBeauty on
SFace) all lowered the zero-rating start, and covering face-recognition space in the picks (B)
lowered the curve. The members simulation was corrected: SCUT cannot show whether other raters'
ratings would help. Onboarding stays at 40 ratings with the adopted crowd prior.

## Generalizing to people the rater has rarely rated (user, 2026-11-13 ~14:50: "I don't think I have
## rated enough black people or men, can you generalize our model without requiring human annotation?")

The user's ratings are mostly of White and East Asian women. Question: how well does the model rank
and place people from groups the rater has barely rated, and can that be improved without new
ratings? This cannot be measured on the user directly (too few such ratings), so it is simulated on
public raters who rated every group. Group labels come only from the datasets' own folders; no
person in the user's pool is labelled by race. Planned 14:55, before running (`group_transfer.py`).

### G1. MEBeauty raters, trained on White and East Asian women only
MEBeauty has 371 individual raters (`generic_scores_all.xlsx`, faces of 6 ethnicities x 2 sexes).
Raters kept: at least 100 ratings of White or East Asian women ("familiar"), 40 of Black people and
100 of men, among the 2,365 faces with codes. Each rater's familiar ratings train the model; it then
scores the familiar faces out-of-fold (5 folds) and every unfamiliar face (averaged over the 5 fold
models). Codes: CLIP L-14 face code + SFace code (the codes both MEBeauty and the pool have), each
column standardized. Ratings model: kernel ridge, strength by leave-one-out from 10^2..10^5.
Crowd score: the adopted SCUT crowd (CLIP + SFace ridges). Methods:
- **current**: rank blend of crowd and ratings model, w = n / (n + 20).
- **E (erase group directions)**: before the ratings model, remove from each code block the
  directions separating the 12 sex x ethnicity group means (11 dimensions, learned from MEBeauty's
  folder labels; no ratings used), so the model cannot score a group as a whole.
- **U (lean on the crowd where ratings say little)**: per person, s = q / q_ref capped at 1, where q
  is the share of the person's code inside what the ratings model learned
  (k' (K + aI)^-1 k / k(x, x)) and q_ref its median over rated people out-of-fold; w = s n / (n + 20).
  Needs no group labels.
- **E + U**.
Measures per rater (rank scale): Spearman on the mixed set (familiar out-of-fold + all unfamiliar;
primary), within familiar, within Black faces, within men, within other women; placement error =
mean |predicted - true mean percentile| over the four unfamiliar groups (Black women, Black men,
other men, other women). Rule: a method is worth building for the pool if the mixed Spearman rises
by 0.02 or more over current, for at least two thirds of raters, and within-familiar Spearman drops
by no more than 0.01. Before any change to the pool model it must also not lower the user's CV by
more than 0.01.

### G2. SCUT raters, trained on women, scored on men
60 SCUT raters; per split (5 splits): faces halved at random; the crowd ridge is fitted on half A to
the mean of the other 59 raters; the rater "rates" 400 random women of half B (familiar) and is
scored on 1,000 random men of half B plus the familiar faces out-of-fold. Same codes, methods (E uses
the MEBeauty directions, as the pool would), measures (within men, placement error for Asian and
White men) and rule.

Results (`results/group_transfer.json`, login node):
- G1, 27 MEBeauty raters. Current model: mixed 0.416; within White / East Asian women 0.448, Black
  faces 0.288, men 0.303, other women 0.429; placement error 0.074. The crowd alone ranks Black faces
  at 0.100 (SCUT has none). E 0.326 (wins 0/27) -> no: the group directions also carry taste.
  U 0.428 (+0.011, wins 19/27, familiar +0.002) -> no. E + U 0.349 -> no.
- G2, 60 SCUT raters x 5 splits. Current 0.665 (men 0.633); E 0.641 -> no; U 0.685 (+0.0198, wins
  300/300, familiar +0.003) -> no, just under the 0.02 bar; E + U 0.662 -> no. Crowd alone 0.692
  (SCUT raters agree with their crowd).
- Reading: the ratings model partly carries over to unrated groups (about two thirds of its
  within-group accuracy); leaning on the crowd where the ratings say little helps a little and
  consistently, but not by the planned margin; removing group directions hurts. The gap for Black
  faces is the crowd's: SCUT has no Black faces, so a crowd with Black faces is the next step.

### G3. MEBeauty v2.1 dating-preference raters (planned 2026-11-14 01:10, before running; `mebeauty_v2.py`)
User approved the download (Hugging Face, gated; the user accepted the terms and downloaded the
`personalized_date` files; kept private on the cluster, never in git). 2,462 faces, 631 raters who
rated both sexes and six ethnicities. Codes as in G1 (CLIP L-14 b82k + SFace, computed on the v2
face images). Faces are split at random in half (seed 7): half A only trains crowd M; everything is
scored on half B. Raters kept: in B, at least 40 ratings of White or East Asian women, 15 of Black
people and 40 of men. Methods: current, E, U, E + U as in G1, plus
- **M (a crowd that has seen Black faces)**: crowd score = mean of the standardized SCUT crowd and a
  MEBeauty crowd (CLIP and SFace ridges fitted on half A to the mean of the other raters, leaving the
  rater out); **M + U**.
Measures as in G1, plus within Black women. Rule as in G1 (mixed +0.02 or more, two thirds of raters,
familiar no worse than -0.01). M can only reach the pool if it also passes the user's own check: the
zero-rating Spearman on the training accounts may not drop by more than 0.01 (MEBeauty's crowd on its
own lowered it to 0.317 from 0.386, so this is expected to fail unless M is limited to where SCUT
knows little).

Result (`results/mebeauty_v2_g3.json`; embed job 12365795, SFace face found in 2,462 of 2,462): 61 raters
kept. Current 0.246 (familiar 0.284, Black 0.175, Black women 0.265, men 0.160). E 0.176 -> no;
U 0.247 (+0.001, 31/61) -> no; E + U 0.194 -> no. **M 0.278 (+0.033, wins 50/61, familiar +0.008) and
M + U 0.314 (+0.069, 51/61, familiar +0.021) pass the simulation rule**; M + U ranks Black people at 0.269
(current 0.175) and Black women at 0.308 (0.265). The SCUT crowd alone ranks Black faces at -0.012.
Caveat (as for SCUT): these raters are scored against their own population's crowd (MEBeauty's crowd
alone reaches 0.353), so the gain is partly that crowd's agreement with its own raters. The user check
decides (`m_crowd_user.py`).
User check (`results/m_crowd_user.json`): MEBeauty's dating crowd alone agrees with this rater at 0.129
(test 0.043); M lowers the zero-rating start to 0.309 from 0.386 (test 0.214 from 0.298) and loses all 20
splits at n = 20 -> **M is not adopted**. As with SCUT, a crowd helps the raters it was drawn from; this
rater's taste is far from MEBeauty's crowd. Whether M helps this rater on Black faces in particular can
only be measured on the fresh batch (added there).

## Owner identity with a stronger face model (user, 2026-11-13 ~18:00: "some of your data shows
## multiple people under one person, you might need a better model to trace identity")

Audit with the existing SFace codes (`owner_audit.py`, counts only): of 3,526 kept photos of 412
people, 105 (3.0%, 87 people) have an owner face that agrees poorly with the person's other photos
(median cosine < 0.30); 32 people have 2+ separate face groups of 2+ photos; 108 people have at
least one such sign. Bodies are almost never wrong: in 12 photos (0.3%) the cut-out body's own nose
and eyes lie away from the owner's face; the whole-photo fallback with 2+ faces happened once. So
the weak point is face identity: SFace's same-person line (0.363) lets look-alikes in.

Plan (`owner_arcface.py`, planned 18:40, before running): every face the owner step's detector
finds (YuNet, same rows) in every photo of all 467 accounts gets an ArcFace R50 code (InsightFace
buffalo_l, WebFace600K; 5-point aligned 112 x 112; user approved the download). ArcFace's
same-person line is set from the data before looking at any account: the 99.9th percentile of
cosines between SFace-kept owner faces of different accounts (the 7 known same-person pairs left
out), i.e. about 1 false match in 1,000 pairs. The owner step is then rerun on ArcFace codes
(unchanged rule: the face most photos agree on, at least 4 photos; each photo keeps its best face
if it matches the owner template). Reported: people whose owner changes, photos SFace kept that
ArcFace rejects, photos ArcFace adds, people who fall below 4 photos. Check before adopting: 20
rejected photos chosen at random are looked at as face crops beside two kept faces of the same
person; ArcFace is adopted for the owner step if at least 15 of the 20 are clearly someone else.
Then the full model and the few-rating curve are rerun on the cleaned photo sets and reported
(not a gate: mixed photo sets are errors, fixed on correctness grounds).

Result (job 12361982 embeds, eval on the login node; `results/owner_arcface.json`): ArcFace line
0.260 (99.9th percentile of 1.69 M cross-account pairs); SFace-kept pairs within a person: median
0.530, 4.0% below the line. Owner a different person under ArcFace: 0 of 412. SFace-kept photos
ArcFace rejects: 37 (26 people); photos ArcFace adds: 123 (103 people); 2 people fall below 4 photos;
7 dropped accounts become usable. Check: of 20 rejected photos about 3 are clearly someone else;
most are the same person in an extreme pose, expression, costume or motion blur, and 8 come from the
two accounts ArcFace drops (blurred video frames). 3 < 15 -> **ArcFace is not adopted** for the owner
step. Face identity is not where people get mixed. The check did show a different problem: the face
crop (2.2 x the face box, from the raw photo) shows neighbours' faces in group photos; counted next
(`cutout_faces.py`).

Count (`cutout_faces.py`, YuNet on the images the models see): cut-outs with 2+ faces 15 photos (0.4%,
15 people, none with 3+); **face crops with 2+ faces 139 photos (3.9%, 93 people, 11 people with 3+)**.
So the mixing is in the face crops, which feed the CLIP / SigLIP face codes and the crowd score.

### Owner-only face crops (planned 2026-11-14 03:35, before running; `face_own.py`)
The same face square is cut from the owner-only cut-out (everyone else already grey), padded grey to a
square, and embedded with the same CLIP L-14 and SigLIP models. Compared with the current crops: full
model CV (as afternoon A), zero-rating start, few-rating curve (same 20 splits). Rule: a correctness
fix, adopted unless full-model CV drops by more than 0.005 or the few-rating mean over n = 5-40 drops by
more than 0.005.
Result (job 12369327; `results/face_own.json`): full model CV 0.6762 vs 0.6695 (+0.007); zero-rating
start 0.361 vs 0.386 (test 0.271 vs 0.298); few-rating 0.384 / 0.426 / 0.471 / 0.506 / 0.586 vs 0.414 /
0.455 / 0.483 / 0.547 / 0.575 (mean over n = 5-40 -0.028, wins at n = 20 9/20) -> **keep the current
crops** (the rule covers both sides). The crowd, learned on studio faces, reads grey-padded crops worse.
Post-hoc observation, not adopted: owner-only crops in the full model alone would be +0.007; that needs
its own decision.

## Photo conditions (planned 2026-11-14 01:25, before running; `photo_conditions.py`)
The user asked whether showing everyone the same way would make ratings fairer. A reconstructed avatar
is ruled out (no avatars of these people; reconstruction is least faithful for darker skin), but the
question can be measured: do photo conditions move the full model's score of the same person? Photo-level
out-of-fold predictions of the adopted full model (same folds as `taste_full.py`); conditions per photo:
log resolution, log face size (eye distance), share of the crop that is the person, 2+ people, and scene
brightness (mean luminance of the photo with every person blanked; skin is never measured, so the
adjustment cannot act on skin tone). Within-person regression of score on conditions (the person held
fixed); adjusted person score = mean photo score minus the effect of the person's average conditions.
Rule: the adjustment goes into the full model if conditions explain at least 5% of the within-person
score variance and Spearman with the user's ratings drops by no more than 0.005. Reported: effects per
condition, people whose rank moves 10+ percentile points.
Result (`results/photo_conditions.json`, 407 people, 3,494 photos): per 1 SD of each condition the same
person's score moves by +0.020 (resolution), -0.029 (face size), -0.024 (person share of the crop), -0.015
(2+ people), +0.015 (scene brightness) rating points; together 1.5% of the within-person variance. The
person-level effect of average conditions has SD 0.031 rating points against a score SD of 0.676;
adjusting changes Spearman by -0.002 and moves nobody by 10+ percentile points -> **no** adjustment
needed: the full model's scores barely depend on how a person was photographed.

## Fresh test batch of new profiles (planned 2026-11-14 01:25, before any is collected)
The user offered to rate more Instagram profiles, with Black people and men in mind. The user chooses
the profiles (a list, or seed accounts whose "similar accounts" fill the batch); no person is labelled
by race by any model. About 60 profiles are collected, cut out and embedded exactly as the pool. Before
the rating page opens, the frozen models score them: the full model trained on all 407 ratings and the
40-rating onboarding model trained on the user's earlier 40 onboarding answers; the scores are saved with
a SHA-256 in this file. After rating, reported once: Spearman on the new batch (a clean test the models
never saw), and placement: the batch's mean predicted percentile against its mean rated percentile within
pool + batch. Also on the new batch, before it joins training: the zero-rating Spearman of the adopted
crowd (SCUT) and of crowd M (SCUT + MEBeauty dating), to see whether a crowd that has seen Black faces
helps this rater where the batch has them (G3 follow-up). Then the batch joins training and the CV is
rerun.

## MEBeauty batch: rating Black people and men without looking for profiles (planned 2026-11-14 07:10,
## before picking; `mebeauty_batch.py`)
The user does not want to search Instagram for profiles but will rate. 60 MEBeauty v2.1 photos (the
dataset's own group labels; no model labels anyone): 15 Black women, 15 Black men, and 6 men each of
the White, East Asian, Hispanic, Indian and Middle Eastern groups, picked by kernel herding on CLIP codes
within each group (photos with no detectable face skipped), shown in a mixed fixed order on a local
page like the stock-face page, plus 6 unannounced repeats for consistency. Before the page opens, these
predictions are frozen (file SHA-256 recorded here): (1) the crowd (SCUT, adopted); (2) crowd M (SCUT +
MEBeauty dating crowd, its ridge trained without these 60 faces); (3) the user's taste from faces alone
(ridge on all 407 rated people's CLIP face, SigLIP face and SFace codes; the full model also needs the
no-clothing cut-out, which these portraits do not have); (4) the onboarding blend of (1) and (3) at
w = 407 / 427. Reported once, with 95% bootstrap intervals: Spearman of each with the user's ratings on
all 60, within Black women and men (30), within Black women (15) and within men (45). Only rankings
within groups are reported; no group's average rating is compared with another's. Reading: (3) and (4)
near their pool level (about 0.6) would mean the taste carries over to these groups; (2) clearly above
(1) within Black faces (intervals not overlapping) would show where crowd M helps this rater, the
condition for a later, label-free way to use it.
Record: 60 picked (15 / 15 / 6 x 5), page with 6 repeats (66 items, each repeat at least 10 places after
its first showing). Predictions frozen 2026-11-14 before the page opened (GPU job 12372951 for the
codes): `results/mebatch_pred.json` SHA-256
`72f5b21b87e9f20bab4c2b8cbd35d00d12c26ed65c49ef77326cd3ab797e5ef5`.
Result (rated 2026-11-14, 60 of 60, none skipped; `results/mebatch_eval.json`): consistency on the 6
repeats 0.69 (pool 0.87). Spearman [95% bootstrap]: your face model 0.30 [0.05, 0.52] on all 60, 0.06
[-0.32, 0.40] within Black women and men, 0.24 [-0.37, 0.72] within Black women, 0.32 [0.03, 0.56]
within men; blend 0.31 / 0.06 / 0.24 / 0.34; SCUT crowd 0.34 / -0.02 / 0.02 / 0.36; crowd M 0.26 / -0.05
/ 0.27 / 0.37. Reading: on these portraits the model ranks men moderately and Black people not at all;
crowd M does not help within Black faces (-0.05 vs -0.02) -> no label-free use of M is pursued.
Follow-up (planned after this result, before running; `mebatch_learn.py`): (a) the face-only model's
pool CV, the fair reference; (b) leave-one-out over the 60: the face model trained on the 407 rated
people plus the other 59 batch photos (batch rows weighted x5, fixed now), within Black faces and men.
Follow-up result (`results/mebatch_learn.json`): (a) face-only model on the pool, CV 0.641; so the batch's
0.30 is far below the pool level: the taste learned from Instagram accounts carries over weakly to these
portraits, and not at all within Black faces. (b) With the other batch ratings in training: all 60 0.298
-> 0.440, Black women and men 0.061 -> 0.196, men 0.321 -> 0.431. The user's own ratings of these groups
teach the model where crowds and transfer did not; with 30 Black faces the within-group number is still
noisy, so more ratings of these groups are the lever.

## MEBeauty batch 2 (user, 2026-11-14: "Yes, set it up"; planned before picking; `mebatch2.py`)
100 new MEBeauty v2.1 photos, none from batch 1: 35 Black women, 35 Black men, 6 men each of the White, East
Asian, Hispanic, Indian and Middle Eastern groups; kernel herding within each group that counts the batch 1
photos as already covered; 8 unannounced repeats. Frozen before the page opens (SHA-256 recorded here): the
face model trained on the pool alone, and on the pool plus the 60 batch 1 ratings (weight x5, as in the
follow-up). Reported once, with 95% bootstrap intervals: Spearman on all, within Black women and men, Black
women, Black men and men. This is the clean test of whether batch 1 taught the model something that carries
over to new faces of these groups (pool + batch 1 above pool alone within Black faces). Afterwards both
batches join the face model's training.
Record: 100 picked (35 / 35 / 6 x 5); codes from GPU job 12373255; predictions frozen before the page
opened: `results/mebatch2_pred.json` SHA-256 `34cd3497e4b9bbbd4cd3a40ee98aae03d8c777702d891775bd7e520433a77d8a`.
Result (rated 2026-11-14, 100 of 100, none skipped; `results/mebatch2_eval.json`): consistency on the 8
repeats 0.90 (batch 1: 0.69). Pool-only face model: all 0.40 [0.22, 0.56], Black women and men 0.34
[0.12, 0.53], Black women 0.23 [-0.12, 0.54], Black men 0.42 [0.12, 0.66], men 0.48 [0.26, 0.64]. Pool +
batch 1 (x5): 0.30 / 0.25 / 0.17 / 0.35 / 0.35 -> **batch 1 did not teach anything that carries over**;
the leave-one-out gain inside batch 1 was not real for new faces. And the pool-only model ranks these
groups far better than batch 1 suggested (0.34 vs 0.06 within Black faces; batch 1 had 30 Black faces,
rated less consistently). Over both batches the taste learned from Instagram carries over to Black faces
at roughly half and to men at roughly three quarters of its pool level (face-only pool CV 0.64).
Follow-up (planned after this result, before running; `mebatch_weights.py`): pool + one batch at weight
0, 0.5, 1, 2 or 5, tested on the other batch, both directions. Portrait ratings join the face model only
at a weight that beats the pool-only model within Black faces in both directions.
Result (`results/mebatch_weights.json`): batch 1 -> batch 2, within Black faces 0.339 (weight 0), 0.306,
0.287, 0.261, 0.248 (weights 0.5-5): worse at every weight. Batch 2 -> batch 1: 0.061 (weight 0) vs -0.026
to -0.079; overall and for men it helps there (0.386 / 0.432 at weight 1 vs 0.298 / 0.321). No weight beats
the pool-only model within Black faces in both directions -> **portrait ratings do not join the face
model**. Summary of both batches: the Instagram-trained taste ranks men at about 0.3-0.5 and Black people
at about 0.1-0.3 on these research portraits; neither public crowds nor portrait ratings improve the
ranking within Black faces. What remains: Instagram ratings of more Black people and men (the pool's own
kind of photo), e.g. profiles the model picks as least covered by the user's ratings.

## No more ratings: face codes for groups the rater rarely rated (user, 2026-11-14: "I am not rating
## anymore, focus on those that don't need me to rate"; planned before running; `face_codes_bench.py`)
The two portrait batches (160 ratings, 100 of Black women and men, 95 of men) become a fixed benchmark
for groups the pool lacks. Face-only taste model as in the batches (RidgeCV on person-level face codes,
trained on the 407 rated pool people only). Variants, fixed now: C0 CLIP + SigLIP + SFace (current);
C1 CLIP + SigLIP; C2 C0 + ViT-H; C3 C0 + DINOv2; C4 CLIP + SigLIP + ArcFace; C5 C0 + ArcFace; C6 SigLIP +
SFace; C7 all six. Measured: pool CV (grouped 5-fold x 3, training accounts), batch 1 (validation) and
batch 2 (test) Spearman on all, within Black women and men, within men. Choice: among variants whose pool
CV is within 0.01 of C0, the one with the highest batch 1 Spearman within Black faces. It is tested once
on batch 2 and goes into the full model's face block only if, on batch 2, within Black faces rises by
0.05 or more over C0, all and men drop by no more than 0.02, and the full model's pool CV (photo-level,
as afternoon A) stays within 0.005 of 0.669.

Result (`results/face_codes_bench.json`): pool CV / batch 1 within Black / batch 2 within Black: C0 0.641 /
0.061 / 0.339; C1 0.626 / 0.026 / 0.310; C2 0.638 / 0.026 / 0.379; C3 0.601 / 0.074 / 0.347; C4 0.642 / 0.055 /
0.343; C5 0.646 / 0.096 / 0.348; C6 0.640 / 0.168 / 0.353; C7 0.621 / -0.009 / 0.411. Chosen on batch 1: C6
(SigLIP + SFace); batch 2 test: within Black +0.014, all -0.001, men +0.021 -> **current codes stay**. (C7, all
six codes, is best on batch 2 but was ineligible: pool CV 0.02 lower.)

## Owner-only face crops in the full model: confirmation (planned before running; `face_own_confirm.py`)
Post-hoc observation (face crops section): owner-only crops gave the full model +0.007. Confirmed on fresh
folds: full model CV with current vs owner-only face crops on 10 new fold seeds (100-109). Adopted for the
full model only (the crowd and onboarding keep the current crops) if the mean gain is +0.005 or more and
owner-only crops are better on 7 or more of the 10 seeds.
Result (job 12374142, `results/face_own_confirm.json`): current 0.6839, owner-only 0.6934 (mean over seeds
100-109, +0.0095), better on 10 of 10 seeds -> **adopted for the full model** (`taste_full.py` reads the
owner-only face codes from `features_faceown.npz`; the crowd and onboarding keep the current crops).

## What the latent space may miss (user, 2026-11-14: "Could our latent space miss something" / "Yes, run
## them"; planned before running; `latent_gaps.py`)
The rater agrees with their own repeats at 0.87, so a perfect model would reach about 0.93 against single
ratings; the full model reaches 0.67 (CV). Five tests on the existing ratings, no new ratings. Base: the
adopted full model (photo-level ridge on L-14 + SigLIP no-clothing and face codes plus SFace, strength
10,000), grouped 5-fold CV on the training accounts, fold seeds 0-2 (as afternoon A).
- **T1 Best photos.** Person score from the same photo predictions as the max, the mean of the top 3 and
  the median, instead of the mean.
- **T2 Style back in.** Add the with-clothing cut-out codes (L-14 + SigLIP `cut`), or the clothing-only
  codes (L-14 `cloth`), to every photo row.
- **T3 Several types.** (a) k-nearest-neighbour regression on person-level codes (cosine; k from 5, 10,
  20, 40 by leave-one-out inside the training fold); (b) rank average of the full model and (a);
  (c) per-cluster ridge (k-means, 3 clusters on PCA-32 of person codes, fitted on the training fold),
  rank-averaged with the full model.
  Rule for T1-T3: replaces the full model if CV reaches 0.689 (0.669 + 0.02).
- **T4 Measurements on the residuals.** Per person (median over photos): face asymmetry, face proportions
  (nose and mouth distance from the eye line, mouth width, all per eye distance) and head roll from the
  five YuNet landmarks; shoulder-to-hip width and leg-to-torso length from the body keypoints where
  visible; hair share of the person; face typicality (SFace distance to the pool mean). Skin colour is
  not measured. Spearman of each with the full model's out-of-fold residuals (rank of rating minus rank
  of prediction); a measurement is "missed by the latent space" if its Holm-adjusted p < 0.05. They join
  the model only if a cross-fitted second stage (ridge on the full model's inner out-of-fold prediction
  plus the measurements) raises CV by 0.02 or more.
- **T5 Outside knowledge.** Out-of-fold Spearman within followed and within not-followed training
  accounts, and the mean rank residual of each group, with bootstrap intervals. Diagnostic only: the
  model stays vision-only.
Results (job 12374141, `results/latent_gaps.json`; base CV 0.6695):
- T1: max 0.643, top 3 0.660, median 0.664 -> **no**; the mean of all photos is best.
- T2: with-clothing codes 0.665, clothing-only codes 0.671 (+0.001) -> **no**; style adds nothing the
  no-clothing and face codes lack.
- T3: kNN 0.515; full model + kNN 0.627; full model + per-cluster ridge 0.643 -> **no**; one direction
  beats local and per-type models (k chosen 20-40).
- T4: no measurement relates to the residuals after Holm correction (largest: shoulder / hip width
  -0.133, face typicality +0.127, both Holm p about 0.2; face geometry |rho| <= 0.02); the second stage
  lowers CV (0.646) -> **no**.
- T5: within followed accounts 0.432 (n = 99), within not-followed 0.580 (n = 206); difference -0.147
  [-0.364, +0.039]; followed people rated slightly higher than their photos predict (mean rank residual
  +0.035 [-0.017, +0.087]). Suggestive of outside knowledge, not significant.
Reading: the probes find nothing the latent space misses. The remaining gap to the rater's consistency is
more likely data (the learning curve is still rising) and things no photo carries (T5 hints at outside
knowledge), than a missing visual property.

## Within-person photo variation (user, 2026-11-15: can the model tell which image of a person the rater
## likes? "Yes, run it"; planned before running; `within_person.py`; no new ratings, training accounts only)
Psychology finds that photos of one person vary in judged attractiveness as much as, or more than, different
people do (Jenkins et al. 2011; Todorov & Porter 2014): an intraclass correlation (ICC: share of photo-score
variance that lies between people) of 0.5 or lower for human judges. Measured here for the models'
photo scores (out-of-fold, grouped 5-fold, seeds 0-2, 305 training accounts, 2,6xx photos):
- ICC(1) of the adopted full model's photo scores (owner-only face crops), and of the crowd's per-photo
  face score (SCUT ridges on each photo's CLIP face code and SFace code).
- Reliability of the within-person ordering: mean over people (3+ photos) of the Spearman between two
  fits' photo scores within the person: seed 0 vs seed 1 of the full model; a CLIP-only vs a SigLIP-only
  photo model (seed 0).
- What moves a person's photo scores: share of within-person variance explained by framing (face or
  close-up / head and shoulders / to the hips / to the knees / full body).
Reading, fixed now: a model ICC well above 0.5 (0.8 or more) means the model treats a person's photos as
near-identical and cannot express the within-person differences people see; image-level liking then needs
image-level labels. A within-person ordering reliability of 0.5 or more across model families means there
is a consistent photo-level signal worth testing against image-level labels later.
Result (job 12393382, `results/within_person.json`; 2,636 photos of 305 people): ICC(1) full model 0.725,
crowd per photo 0.416. Typical spread of one person's photo scores 0.345 rating points against 0.614 between
people's means. Within-person ordering: full model seed 0 vs seed 1 0.847; CLIP-only vs SigLIP-only 0.544
(305 people). Framing explains 1.9% of the within-person variance (photo conditions 1.5%, earlier).
Reading: 0.725 < 0.8, so the model does tell one person's photos apart (about a third of its score
variance lies within people), less than a generic crowd score or human judges do; and two different image
models agree on the order of a person's photos at 0.54 >= 0.5, so there is a consistent photo-level signal,
not explained by framing or photo conditions. Whether that order matches which photos the rater likes is
untested: it needs image-level labels.

## Photo pairs: does the model pick the photo the rater prefers? (user, 2026-11-15; planned before picking;
## `photo_pairs.py`)
60 people drawn at random (seed 20261115) from the training accounts with 4+ kept photos; one pair of that
person's photos each. Pairs 1-40: the full model's highest- vs lowest-scoring photo of the person; pairs
41-60: two photos drawn at random. Photo scores: the adopted full model (owner-only face crops), out-of-fold
(the person's own rating never used), mean over fold seeds 0-2; the crowd's per-photo face score is frozen
too. Shown as on the rating page (the person cut out on grey), left / right at random, in random order, plus
6 unannounced repeats with sides swapped. The rater picks the photo they like more (or skips). Predictions
frozen before the page opens (SHA-256 recorded here). Reported once: share of pairs where the rater picks
the model's photo, overall and for each pair type, with a one-sided binomial test against 50%; the same for
the crowd's pick; consistency on the repeats. Reading, fixed now: the model's photo-level signal predicts the
rater's image-level preference if overall agreement is above 50% with p < 0.05.
Record: pairs picked and predictions frozen 2026-11-15 before the page opened (job 12393457):
`results/photo_pairs_pred.json` SHA-256 `dfae5209e091dcd0394d3efb06d9694df9fde527c64b67af85191295222fae4c`.
Result (`results/photo_pairs_eval.json`; 40 of 60 pairs answered, 20 skipped as "can't choose"; the 2
answered repeats matched): the model's photo was picked in 25 of 40 pairs (62%, one-sided p = 0.077) ->
**not shown** by the rule (overall p < 0.05). By pair type: the model's best vs worst photo of a person 21 of
27 (78%, p = 0.003); random pairs 4 of 13 (31%). The crowd's per-photo pick: 23 of 40 (58%, p = 0.21); 16 of
27 and 7 of 13. Among best-vs-worst pairs, a larger model gap went with more agreement (Spearman 0.49).
Reading: when the model separates a person's photos clearly, the rater mostly agrees (78%), and the taste
model does better than a generic attractiveness model there (59%); between photos it scores alike it says
nothing useful. A third of the pairs were skipped, which cut the test's power; a confirmation would need
more pairs of the best-vs-worst kind.

## Changes

- 2026-11-12, before any photo was downloaded or rated: pool size set to about 400 accounts
  (the user's 138 public follows + about 260 suggested accounts they do not follow), instead of
  "a third more". Reason: with about 210 accounts the test set would hold about 50, too few for
  a true correlation of 0.5 to clear the rule's lower bound of 0.30. User's choice ("About 400").
- 2026-11-12, same stage: 330 suggested accounts are collected (not 263), because some accounts
  will drop out at the owner step (brands, group pages); the aim is about 400 after it. Picking
  rule and seed: `pick_contrast.py` (2 per follow in a seeded random order, then 1 more per follow).
  Photos: the profile's header text (bio) is kept so a stated height can be read.
- 2026-11-12, during collection, before any rating: photos per account are the first 12 posts the
  profile page shows (24 for the first 13 accounts, where a second page loaded). In the app's
  browser pane, which is usually hidden, Instagram does not load a profile's second page, so 24
  per account was not reachable without many more page loads. The pilot found the owner on 10-11
  of 12 photos per account.
- 2026-11-13, 11:10: crowd prior for onboarding changed from the SCUT crowd model on CLIP face
  codes to the mean of that and a SCUT crowd model on SFace face-recognition codes, by the rule
  pre-registered at 09:36 ("Crowd model on face-recognition codes"). `taste_onboard.py` and the
  onboarding page use it; the full model is unchanged.
- 2026-11-13, 13:20: full model (all ratings) changed from account-level ridge on L-14 + SigLIP
  (CV 0.636) to photo-level ridge on L-14 + SigLIP photo codes plus the account's SFace code (CV
  0.669, strength 10,000), by the rule pre-registered at 13:00 ("Afternoon exploration", A).
  `taste_full.py` implements it.
- 2026-11-13, 13:20: correction to the reading of "Shared taste dimensions": on SCUT the gain
  came from anchoring on the crowd, not from other raters' directions (see the members
  simulation). The README's "Next step" section was rewritten accordingly.
- 2026-11-14: the full model uses owner-only face crops (confirmed on 10 fresh fold seeds, +0.0095,
  10/10); crowd and onboarding unchanged.
