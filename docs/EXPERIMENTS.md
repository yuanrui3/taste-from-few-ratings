# Experiments

All numbers are for the first rater unless a dataset is named (SCUT-FBP5500: 60 raters of studio faces;
MEBeauty: raters of portraits from six ethnic groups). "Rule" means the decision rule written in the
protocol before the test ran; a test that missed its rule was not adopted, however close.

One-off studies live in `taste/experiments/`. They import from the pipeline folder and write
`results/<name>.json`. Each has a section in [`PROTOCOL.md`](PROTOCOL.md), where its decision rule was written before it ran.

| Script | Question | Outcome |
| --- | --- | --- |
| `probe_search.py` (pipeline) | Which people to rate first? | Kernel herding |
| `combine_search.py` | Blend a prior with few ratings: how? | Rank blend |
| `taste_explain.py`, `taste_latent.py` | Which describable traits go with the ratings? A map of the latent space | Exploratory; see [`PROTOCOL.md`](PROTOCOL.md) |
| `attr_fewshot.py` | Learn from described attributes instead? | No (0.41 vs 0.60) |
| `fitness.py` | A fitness score from body crops | Measures visible muscle definition; did not track the ratings; not used |
| `style_confound.py` | Does it reward professional photography? | No |
| `backbone_eval.py` | Larger image models? | SigLIP helps with many ratings (+0.04), not with few |
| `scut_eval.py` | Do the methods work for 60 other raters? | Herding 58/60; crowd prior 60/60 |
| `crowd_prior_eval.py`, `mebeauty_prior.py`, `followups.py` | Which crowd prior? | SCUT, all faces, ViT-L-14 |
| `probe_prior.py` | With a crowd prior, rate different people first? | No, herding stays |
| `adaptive_blend.py` | A per-person blend weight? | Helps SCUT raters, hurts the first rater; not adopted |
| `stock_probe.py` | Can stock faces replace pool photos? | No: studio faces show too little of the person |
| `improve_r1.py` | Photo-level training, ensembles, kernel ridge; zero-shot "attractive" prompts as a prior; transfer ridge; low-dimensional model | None passed; photo-level training +0.016 with all ratings; prompts nearly useless (0.05) |
| `switch_check.py` | Larger codes once 40+ ratings exist? | Not at 40; better from about 80 (15/20, 17/20 at 120) |
| `identity_space.py` | Self-supervised: keep only directions that are stable across one person's photos? | No: they track facial identity, not taste (0.38 vs 0.46 at 20 ratings) |
| `active_next.py` | After 20 ratings, who to rate next (uncertainty, disagreement, top-predicted)? | Keep herding order (0.53 / 0.56 at 40 / 60) |
| `prior_typical.py` | Free prior: how average a face is | 0.21 with no ratings or outside data, but adds nothing to the crowd model |
| `prior_raters.py` | Weight SCUT's 60 raters by agreement with the first ratings | +0.005, consistent but tiny; sharper weights worse |
| `block_weight.py` | Weight face vs body codes differently | Equal weighting stays; more body weight is worse |
| `photo_fewshot.py` | Few-rating model trained on single photos | Same up to 40 ratings; better at 80 (0.58 vs 0.57) |
| `coarse_ratings.py` | Coarser answers (like / pass; low / mid / high) to save effort? | Like / pass needs 4x the answers; keep 0-10 |
| `flip_tta.py` | Average each photo with its mirror image | No change |
| `residuals.py` | Where does the full model go wrong? | Mid-rated people and polished photos; no data-quality fault |
| `prior_domain.py` | Align the crowd model's studio faces to Instagram faces | Best variant (kernel ridge, each set standardized separately): 0.38 vs 0.35 with no ratings, 0.48 vs 0.46 at 20; one split short of the rule |
| `prior_confirm.py` | Does that aligned crowd model hold on the 99 held-out accounts? | Not confirmed (+0.023, interval includes 0); current crowd model stays |
| `laprls.py` | Semi-supervised: spread few ratings along the pool's neighbour graph | No gain (0.464 vs 0.459 at 20) |
| `prior_mebeauty_check.py` | Does the crowd-model alignment help on another face set? | No (0.586 vs 0.611): the lead is closed |
| `improve_r2.py` | Full model from single photos with three image models, or an ensemble | 0.654 / 0.653 vs 0.636: about +0.017, just under the +0.02 rule |
| `multi_rater_scut.py` | Would other raters' taste directions help a new rater? (SCUT simulation) | Looked like yes (46/60 raters), but the gain came from anchoring on the crowd, not the other raters (see `members_sim.py`) |
| `multi_rater_user.py` | Do SCUT's taste directions carry over to the first rater? | No (0.378 vs 0.459): directions must come from raters of the same photos |
| `anchored_residual.py` | Start from the crowd score and learn only the rater's difference | Worse for the first rater (0.43 vs 0.46 at 20); better for 58/60 SCUT raters, who agree with their crowd |
| `sface_input.py` | Face-recognition codes (SFace, 128 numbers) as an extra input | Alone 0.566; small consistent gains (+0.010 full, +0.006 few, 17/20 splits), under the rule |
| `afternoon_ab.py` | Full model from single photos plus face-recognition codes; onboarding picks that cover face codes too | **Full model adopted**: 0.669 vs 0.636 (held-out look 0.723). Picks: no |
| `facenet_aligned.py` | FaceNet on eye-aligned faces as a third crowd code | No: only SFace carries the crowd's taste over |
| `members_sim.py` | How many members, rating how many people, would help a new rater? (SCUT) | Every size "helps", but no more than the SCUT crowd alone: on SCUT the members add nothing measurable; it cannot answer the question for raters who differ from the crowd |
| `full_test_look.py` | Adopted full model on the held-out accounts (labelled look) | 0.723 [0.60, 0.81] |
| `mebeauty_sface.py` | A second crowd (MEBeauty) read through face-recognition codes | No: lowers the zero-rating start (0.317 vs 0.386) |
| `push_down.py` | Fewer than 40 ratings: re-tuned blend weight, SFace codes in the ratings model, FaceNet as a third crowd code | None passed. SFace input reaches the 40-rating score at about 35 (consistent, 18/20, but under the rule); FaceNet on unaligned faces fails |
| `sface_crowd.py` | Crowd model on face-recognition codes, which ignore lighting and make-up | **Adopted.** 0.39 vs 0.35 with no ratings (held-out accounts 0.30 vs 0.19); 0.48 vs 0.46 at 20 ratings, better in 18/20 splits |
| `group_transfer.py` | Groups the rater barely rated (Black people, men): trained on White and East Asian women only, how well are the others ranked? Remove group directions (E), lean on the crowd where ratings say little (U). MEBeauty and SCUT raters | The ratings model carries over at about two thirds strength (Black 0.29, men 0.30 vs 0.45). E hurts; U helps consistently (300/300 SCUT, 19/27 MEBeauty) but by +0.011 / +0.020, just under the rule |
| `mebeauty_v2.py` | Same test with 61 MEBeauty v2.1 dating-preference raters, plus a crowd that has seen Black faces (M) | M and M + U pass for those raters (0.278 / 0.314 vs 0.246; Black people 0.269 vs 0.175) |
| `m_crowd_user.py` | Crowd M on the first rater | No: MEBeauty's crowd agrees with this rater at 0.13, so M lowers the zero-rating start (0.309 vs 0.386). A crowd helps the raters it was drawn from |
| `photo_conditions.py` | Do resolution, face size, framing, group photos or scene brightness move the same person's score? | Barely: 1.5% of within-person variance; adjusting moves no one 10+ percentile points. No adjustment needed |
| `owner_audit.py`, `owner_arcface.py`, `cutout_faces.py` | Are people mixed up under one account? SFace audit, a stronger face model (ArcFace R50), faces counted in the images the models see | Identity is fine (ArcFace changes no owner; most of its rejections are the same person). The mix-up is in face crops of group photos: 3.9% show a neighbour's face |
| `face_own.py`, `face_own_confirm.py` | Face crops cut from the owner-only cut-out, so neighbours are grey | Few-rating side: no (the crowd start drops, 0.361 vs 0.386). **Full model: adopted** after confirmation on 10 fresh fold seeds (+0.0095, 10/10) |
| `mebeauty_batch.py`, `mebatch_learn.py` | 60 research portraits (MEBeauty: Black women and men, men of five other groups) rated by the first rater, predictions frozen before rating | Clean test: 0.30 overall, 0.06 within Black faces (30 faces; repeats agree 0.69), 0.32 within men (pool level 0.64); no crowd does better. The leave-one-out gain from these ratings did not hold on new faces (batch 2) |
| `mebatch2.py`, `mebatch_weights.py` | 100 more portraits (70 Black women and men), predictions frozen before rating; portrait ratings added at weights 0.5-5, tested across batches | Clean test: 0.40 overall, 0.34 within Black faces, 0.48 within men (repeats agree 0.90). Adding batch 1 makes it worse (0.25); no weight helps within Black faces in both directions: portrait ratings are not used |
| `face_codes_bench.py` | Which face codes rank the rarely rated groups best (CLIP, SigLIP, SFace, ViT-H, DINOv2, ArcFace), chosen on portrait batch 1, tested once on batch 2? | No change: the chosen codes (SigLIP + SFace) gain +0.014 within Black faces on batch 2, under the +0.05 rule |
| `latent_gaps.py` | Does the latent space miss something? Best-photo aggregation, style back in, several "types", measurements on the residuals, followed vs not followed | Nothing found: mean of photos, no clothing, one direction all stay best; no face or body measurement explains the errors. Followed people are predicted worse (0.43 vs 0.58), a hint of outside knowledge, not significant |

## Open question: more raters

With one rater, the only change that improved the few-rating method was the face-recognition
crowd model. Whether other members' ratings of the same pool would help a new rater is open.
The SCUT simulations cannot answer it: SCUT's raters agree so closely with their crowd that the
crowd alone predicts them at 0.72, and every member-based model only matched that. A test needs
real raters who, like the first one, differ from the crowd: several members rate the onboarding
page's 40 people (the page's "Save my ratings" button writes a small file with opaque ids and
numbers only), and each member is then left out in turn as the "new rater".

