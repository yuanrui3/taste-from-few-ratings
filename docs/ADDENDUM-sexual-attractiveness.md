# Addendum: sexual implication and sexual attractiveness, per photo

Added 2026-11-16. The main study predicts one rater's appearance rating of a person: one number per
account, judged from all of their photos. This addendum asks the same rater two questions about
**single photos** of the same public accounts, 0-10 each:

- **Sexual implication**: "How much does this photo have sexual implications to you?" The rater's
  own definition: how much the viewer thinks the photo has sexual implications, a perceived
  judgement rather than a measure of skin.
- **Sexual attractiveness**: "How sexually attractive is this person to you in this photo?"

Both differ from person to person; the aim is to model one person's judgement, not a crowd's. As in
the main study, every rule was written before the ratings it judges and each test was scored once.
The photos, the ratings and the list of accounts stay on the rater's machines; this repository holds
only code and aggregate results. With one rater, nothing here shows how people differ.

## Results in brief

| | Round 1 test: 58 people, 116 photos | Round 2 test: 41 new people, 122 photos |
| --- | --- | --- |
| Sexual implication | **Passes.** Model 0.835 [0.790, 0.867]; exposure stand-in alone 0.817; gain +0.019 [0.003, 0.035]; a person's higher-rated photo ranked higher in 40 of 41 pairs | Not retested by rule. Round 1's model on new photos, including ones between the extremes: 0.784; stand-in 0.766 |
| Sexual attractiveness | **Does not pass** (0.488 < 0.50). Beats the stand-in (0.316); pairs 31 of 40. The rater's earlier appearance rating of the person alone: 0.599 | **Does not pass.** Appearance rating + image codes, predictions frozen before rating: 0.708 [0.568, 0.812], but not shown to beat the appearance rating alone (0.652; +0.056 [-0.024, 0.145]). Within a person: right for 20 people, wrong for 5 (p = 0.002) |

Spearman correlations with the rater's answers; intervals are 95% from a bootstrap over people. The
stand-in is the mean percentile rank of five open-source scores of how much body a photo shows.

What it says, for this rater:
- Sexual implication is almost entirely how much body the photo shows. Open-source exposure scorers
  already track it closely; image models add a small but measurable amount.
- Sexual attractiveness depends mostly on who the person is (the earlier appearance rating predicts
  it better than any photo-only model) and secondly on the photo, mostly how revealing it is. A
  model combining the two orders one person's photos correctly for 4 people in 5, something a
  single per-person rating cannot do, but across all photos its gain was too small to show with 41
  people.
- The rater was consistent: on re-shown photos, 103 of 104 answers matched the first one or were 1
  point off (re-rating Spearman 0.74-0.98), but used a narrow band (mostly 5-7), which limits every
  correlation.

Safeguards: only verified public accounts and accounts Instagram suggested from them; two
apparent-age screens (people set aside, single photos dropped), and a "may be under 18" button on
the rating page that removes the person; one-person photos only; no images, account names or
ratings are published.

## Code

| File | What it does |
| --- | --- |
| `taste/experiments/suggest_range.py` | Scores every owner photo with five open-source scorers (LAION CLIP-based NSFW detector, `LukeJacob2023/nsfw-image-detector`, NudeNet exposed parts, a zero-shot CLIP probe, bare-skin share from the clothes parser) and an apparent-age estimate; reports how much range the photos have |
| `taste/experiments/suggest_pack.py` | Second age screen, picks each person's least and most revealing photo, freezes the set (SHA-256), writes the rating page's files |
| `taste/suggest_rate.html`, `taste/suggest_server.py` | Two-question rating page (keys 0-9, T = 10; skip; "may be under 18"; back) and its small local server |
| `taste/experiments/suggest_fit.py` | Round 1: ridge models picked by cross-validation over training people, scored once on the test people |
| `taste/experiments/suggest_fresh.py` | Round 2: picks people never shown before, fits the combined model on round 1 and freezes its predictions (`pick`), then scores them once (`eval`) |

Extra dependencies for the addendum: `nudenet==3.4.2` (install with `--no-deps` next to
`onnxruntime-gpu`; the script builds NudeNet's session itself) and InsightFace's `buffalo_l`
`genderage.onnx` under `--models/insightface/buffalo_l/` (non-commercial research only).

---

The sections below are the protocol as written in the lab notebook, including when each part was
written. Plans come first, then the record made before the rating page opened, then the result.

## Sexual implication and sexual attractiveness, per photo (planned 2026-11-16, before any rating)

Asked by the user on 2026-11-16: reuse the saved photos to rate "sexual exposure" and "sexual
attractiveness", and infer one person's opinion on both. Their definition: "sexual exposure
basically means how much does the viewer think this photo has sexual implications". Both are
judgements of one photo, not of an account, so the unit here is the photo.

**Pool.** 290 accounts with 2,486 owner photos: verified accounts, and unverified accounts that
Instagram suggested from a verified account (`split/suggest_pool.json`). Follower counts were never
saved, so "verified" stands in for "public figure"; 120 unverified accounts and two people from
a separate study are left out. Two apparent-age screens set people aside: InsightFace's age
estimate (median under 21) and a zero-shot check with CLIP (median chance of "child or teenager"
0.5 or more; single photos at 0.7 or more are dropped). Neither is proof of age, so the page also
has a "may be under 18" button that removes the person.

**Stand-in score** (`experiments/suggest_range.py`, job 12397647, no ratings): mean percentile rank
of five scores of the owner's crop (LAION NSFW detector, a "sexy" classifier, NudeNet exposed
parts, a zero-shot probe, bare-skin share). Checked by eye, it orders photos by how much body is
shown. It is used to pick photos and as the baseline, never as the answer.

**Photos rated** (`experiments/suggest_pack.py`, seed 20261116). Only photos with one person in
them. For each person with two or more such photos: their lowest and their highest photo by the
stand-in score. All test people of the frozen split (70) and 130 training people drawn at random;
the other training people are a reserve. About 400 photos; 40 of them come back later in the
session, unannounced, for consistency. Random order. The page shows the owner's crop from the
original photo (surroundings inside the box stay, since they are part of what a photo implies),
under an opaque id, with no handle and no score.

**Questions**, whole numbers 0-10 each:
- Sexual implication: how much does this photo have sexual implications to you? (0 none, 10 very strong)
- Sexual attractiveness: how sexually attractive is this person to you in this photo?

**Split.** By person, the frozen split of this protocol (test people were never used for model
choices). The file `split/suggest_set.json` is written and its SHA-256 recorded below before the
page opens.

**Record** (2026-11-16, job 12397870, before the page opened): the age screens set 55 people aside,
all by the zero-shot check (the InsightFace estimate flagged nobody; lowest median 23.3), and
dropped 312 single photos. The two screens disagree this much because neither is reliable; the rule
above was kept as written since it errs on the cautious side. Rated set: 380 photos of 190 people
(test 120 photos of 60 people, training 260 of 130), 420 screens with the 40 repeats (each at least
60 screens after its first showing), 31 training people in reserve. Photos per fifth of the
stand-in score: 102 / 58 / 68 / 58 / 94. `split/suggest_set.json` SHA-256
`86c0fa0526b5b372f0d6bcc3514c205fa9cee980885b4f4c741dde3e6f0aa95e`. Page: `suggest_rate.html` served
by `suggest_server.py`; answers go to a private file outside this repository.

**Model** (after the ratings; choices by 5-fold cross-validation over training people only): ridge
regression from image codes of the rated photo to each rating. Candidates fixed now: (a) OpenAI
CLIP L-14 code of the shown crop; (b) plus the pipeline's L-14 and SigLIP photo codes; (c) plus
the five stand-in scores as inputs.

**Pass rule** (fixed now; each scale separately; test people's photos, first showing only; scored
once; intervals from a bootstrap over people):
1. Spearman correlation between predicted and actual ratings at least 0.50, lower end of the 95%
   interval at least 0.30.
2. More than exposure: beats the stand-in score; the 95% interval of the difference lies above 0.
3. Within a person: among test people whose two photos got different ratings, the model orders the
   pair correctly more often than chance (one-sided binomial test, p < 0.05).

Also reported, not part of the rule: re-rating consistency; the correlation between the two scales;
each single stand-in score; the person's earlier appearance rating used alone as a prediction of
sexual attractiveness; how often the stand-in score orders a person's pair correctly; women and men
separately. Known limits: one rater, so nothing here shows that people differ; and choosing each
person's two extreme photos widens the range, which raises every correlation, the baseline's too.

**Result** (2026-11-16, `experiments/suggest_fit.py`, `results/suggest_eval.json`; scored once). All
420 screens answered (no skips, no "may be under 18"); 260 training photos of 130 people; test 116
photos of 58 people (2 same-person accounts dropped, see "Change made before fitting" below). The answers sit in a narrow band:
sexual implication 5 or 6 on 321 of 380 photos (range 2-8), sexual attractiveness 4-7 on 366 (range
3-8). Re-rating (40 repeats): implication Spearman 0.74 (30 identical, 39 within 1 point),
attractiveness 0.87 (29 identical, all 40 within 1). The two scales correlate 0.43.
- **Sexual implication: passes.** Chosen by CV: candidate (c), codes plus the five stand-in scores
  (CV 0.766; (a) 0.728, (b) 0.765; stand-in alone 0.765). Test Spearman 0.835 [0.790, 0.867]; the
  stand-in score 0.817 [0.764, 0.856]; difference +0.019 [0.003, 0.035]; the higher-rated photo of a
  person ranked higher in 40 of 41 pairs (p < 1e-10). Rules 1, 2, 3 met.
- **Sexual attractiveness: does not pass** (rule 1). Chosen by CV: candidate (b), the three image
  codes (CV 0.494; (a) 0.488, (c) 0.485; stand-in 0.369). Test Spearman 0.488 [0.335, 0.624], under
  0.50 (the lower end, 0.335, clears 0.30); the stand-in 0.316 [0.169, 0.467]; difference +0.173
  [0.014, 0.340] (rule 2 met); pairs 31 of 40 (p = 0.0003; rule 3 met).
- Also reported. Single stand-in scores on the test, implication / attractiveness: "sexy" classifier
  0.78 / 0.32, bare-skin share 0.70 / 0.30, LAION detector 0.68 / 0.19, NudeNet area 0.65 / 0.32,
  zero-shot probe 0.64 / 0.19. The person's earlier appearance rating alone: attractiveness 0.599
  [0.442, 0.720] (higher than the model; difference model - appearance -0.11 [-0.29, 0.10]),
  implication -0.10. Over all 190 people the more revealing photo (by the stand-in) got the higher
  implication rating in 116 of 123 pairs with different answers and the higher attractiveness rating
  in 97 of 118. Women 106 test photos: implication 0.835, attractiveness 0.538; men 10 photos (5
  people), too few to read.
Reading: in this set the rater's sense of sexual implication is almost entirely how much body the
photo shows; the image models add a small but real amount on top of the stand-in score (+0.02), and
the scale was used nearly as two points (5 for a person's covered photo, 6 for their revealing one),
so the test mostly checks that split. Sexual attractiveness depends mostly on who the person is
(their earlier appearance rating predicts it better than any photo model here) and secondly on how
revealing the photo is (the more revealing photo was rated higher 82% of the time). A model that
combines the two was not a candidate, so it is untested; testing it needs photos not used here.

## Sexual attractiveness, round 2: who the person is plus the photo (planned 2026-11-16, 22:10, before picking photos)

The user, 2026-11-16: "Yes, set it up" (a combined model tested on fresh photos). Round 1's test
people are used up, so this round tests on people never shown in round 1.

**Fresh people.** Every pool person not shown in round 1, not set aside by the age screens, not the
same person as a round-1 account (`same_person.json`), with an earlier appearance rating and at
least one eligible photo (one person in it; zero-shot "child or teenager" chance under 0.7, as in
round 1): the 31 reserve training people and any of the 14 other pool people who qualify.

**Photos.** Up to 4 per person: the lowest and the highest by the stand-in score, plus up to 2 more
drawn at random (seed 202611162), so photos between the extremes are rated too. Same page, same two
questions, same crop as round 1; 12 photos come back at least 30 screens later. Random order.

**Models** (fit on all 380 round-1 photos, first showings, 190 people; round 1's test is finished,
so its photos now count as training):
- Baseline: the person's earlier appearance rating alone (the same 0-10 rating the appearance study
  used; one number per person).
- B: linear regression of sexual attractiveness on the appearance rating and the stand-in score.
- C: ridge on the appearance rating (as its own block) and the three image codes of round 1's
  candidate (b), blocks standardised and scaled to equal weight; strength from round 1's grid.
- B or C is picked by grouped 5-fold CV over the 190 people (fold seeds 0-2, mean Spearman).
Predictions for the fresh photos are written to `results/suggest_fresh_pred.json` (opaque ids only)
and its SHA-256 recorded here before the page opens.

**Pass rule** (fresh people, first showings, scored once; intervals from a bootstrap over people):
1. Spearman between predicted and rated sexual attractiveness at least 0.50, lower end of the 95%
   interval at least 0.30.
2. More than who the person is: beats the appearance rating alone; the 95% interval of the
   difference lies above 0.
3. Within a person: for each person whose photos got at least two different ratings, the share of
   their differently-rated photo pairs that the model orders correctly; people above 0.5 count as
   right, below 0.5 as wrong (exactly 0.5 left out); one-sided sign test, p < 0.05.

Also reported, not part of the rule: the stand-in score alone; round 1's image-only attractiveness
model refit on all 380 photos; round 1's sexual implication model refit on all 380 photos, scored on
the fresh photos (a second check of that pass, now with photos between the extremes); re-rating
consistency; the spread of answers. Known limits: about 35 people, so intervals are wide; if the
reserve is all women (the round-1 range check suggests so), nothing here speaks for men; one rater.

**Record** (2026-11-16, 22:20, job 12406026, `experiments/suggest_fresh.py pick`, before the page
opened): 41 fresh people (31 reserve + 10 of the other pool people; 1 looks male), 122 photos (41
lowest, 31 highest, 50 between; 10 people have one eligible photo), 134 screens with the 12 repeats.
Round-1 CV: B 0.692, C 0.704 (strength 10), appearance rating alone 0.609, stand-in alone 0.348 ->
**C** chosen. B's weights, for reference: 2.245 + 0.460 x appearance + 1.554 x stand-in.
`results/suggest_fresh_pred.json` SHA-256
`ab83f7bfb6b0454b1c94d8e9c899a3e7ff5cd9ee44f40fcf597e9cf74133f851`; private map
`split/suggest_set2.json`. Page: as round 1.

**Result** (2026-11-16, `suggest_fresh.py eval`, job 12406734, `results/suggest_fresh_eval.json`;
predictions file checked against the SHA-256 above; scored once). All 134 screens answered, none
skipped or flagged; 122 photos of 41 people. Re-rating (12 repeats): attractiveness Spearman 0.98,
implication 0.92 (9 of 12 identical on each, all within 1 point).
- **Sexual attractiveness, model C: does not pass** (rule 2). Spearman 0.708 [0.568, 0.812] (rule 1
  met); the appearance rating alone 0.652 [0.498, 0.780]; difference +0.056 [-0.024, 0.145], so not
  shown to beat it (rule 2 not met); within a person, right for 20 people and wrong for 5 (p = 0.002;
  rule 3 met).
- Also reported. Stand-in score alone 0.457 [0.305, 0.602], within a person 22 right / 6 wrong; round
  1's image-only model 0.648 [0.495, 0.766], 21 / 5. Round 1's sexual implication model on these
  fresh photos, including ones between the extremes: 0.784 [0.702, 0.842], stand-in 0.766, within a
  person 19 right / 1 wrong. The two scales correlate 0.59. Answers again sat in a narrow band
  (attractiveness 4-8 on 120 of 122, implication 5 or 6 on 101).
Reading: on people it never saw, the combined model tracks the rater's sexual-attractiveness ratings
at 0.71, the best of anything tried, and gets the order of one person's photos right for 4 people in
5, which the appearance rating cannot do (it is one number per person). But across all photos it is
not reliably better than the appearance rating alone: most of the signal is who the person is, and
the photo-to-photo part (mostly how revealing the photo is: the stand-in alone orders a person's
photos about as well) is too small a share of the variance for 41 people to show the gain. Every
eligible pool person has now been rated, so a larger test needs new accounts.

## Change made before fitting (round 1)

- 2026-11-16, 15:50, sexual implication / attractiveness: details the plan left open, fixed after
  the ratings came in but before any model was fitted or any rating was set against any score (only
  the counts of each answer were looked at: all 420 screens answered, no skips, no "may be under
  18"). (1) Test people: the 2 test accounts that `same_person.json` marks as the same person as a
  training account are dropped from the test, as in the main protocol (`test_after`): 58 test people,
  116 photos. (2) "The pipeline's L-14 and SigLIP photo codes" in candidate (b) are the `cut` variant
  (the owner cut out, clothing kept), not the full model's `nocloth` one, which greys out the clothing
  this question is about. (3) Ridge details: each block of inputs (a code, or the five stand-in scores
  as percentile ranks over the whole pool) is standardised and scaled to equal total weight; the
  strength is picked from a fixed grid together with the candidate, by the mean Spearman over fold
  seeds 0-2 (folds keep accounts of one person together); training targets are first showings.
  `experiments/suggest_fit.py`.
