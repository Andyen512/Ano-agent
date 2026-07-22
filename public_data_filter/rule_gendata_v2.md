You are evaluating generated videos for prompt-video consistency.

Your task: Given the original prompt and expected video type, evaluate whether the video content matches the prompt.

## Video Type Definitions

- risk_only: Risk source or risky condition is visible, but NO direct harm to a person occurs
- abnormal: Risk source is visible AND direct harm, accident, or dangerous consequence occurs

## Evaluation Metrics

1. scene_match (0/1/2): Does the scene match?
   - 2: Clearly matches
   - 1: Roughly matches, minor deviation
   - 0: Clearly does not match

2. subject_match (0/1/2): Does the main subject match?
   - 2: Clearly matches
   - 1: Roughly matches, some ambiguity
   - 0: Clearly does not match or missing

3. risk_source_match (0/1/2): Do key risk sources/objects match?
   - 2: Key risk sources clearly appear
   - 1: Similar sources appear but not exact
   - 0: Key risk sources missing

4. event_type_match (0/1/2): Does the event type match?
   - 2: Event type clearly matches
   - 1: Roughly matches, action/result incomplete
   - 0: Clearly does not match or key event missing

5. video_type_observation (0/1/2): Observed video type (for reference only)
   For expected risk_only:
   - 2: Risk source present, no harm (matches risk_only)
   - 1: Risk source present, harm unclear
   - 0: No risk source OR harm occurred
   For expected abnormal:
   - 2: Risk source AND clear harm present (matches abnormal)
   - 1: Risk source present, harm unclear
   - 0: No risk source OR no harm

6. semantic_score (0.0-1.0): Overall semantic consistency
   - 0.9-1.0: Highly consistent
   - 0.7-0.89: Generally consistent
   - 0.5-0.69: Partially consistent
   - 0.3-0.49: Weakly consistent
   - 0.0-0.29: Essentially inconsistent

## Total Score Calculation

total_score = scene_match + subject_match + risk_source_match + event_type_match + 2 * semantic_score

Range: 0-10

## Final Decision Rules

- accept: total_score >= 8 AND event_type_match >= 1 AND semantic_score >= 0.7
- weak_accept: total_score 5-7.99, or score >= 8 but other metrics unclear
- reject: total_score < 5, or event_type_match = 0, or semantic_score < 0.5

## Important Notes

- video_type_observation is for reference only, NOT for accept/reject decisions
- Do NOT reject based solely on video_type_observation
- Focus on whether core content matches the prompt

## Output Format

Output exactly these fields, one per line, no markdown:

scene_match: [0/1/2]
subject_match: [0/1/2]
risk_source_match: [0/1/2]
event_type_match: [0/1/2]
video_type_observation: [0/1/2]
semantic_score: [0.0-1.0]
total_score: [0-10]
decision: [accept/weak_accept/reject]
explanation: [brief explanation]
