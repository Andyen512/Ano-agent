下面是完整整合后的 Codex prompt：

```text
请帮我实现生成视频的自动评估模块。目前视频分为三类：

- normal：正常视频，不进行评估，直接跳过。
- risk_only：仅出现风险源或风险条件，但目前没有对人造成伤害。
- abnormal：出现风险源，并且已经发生了对人的伤害行为或直接危险后果。

当前阶段只评估 Prompt-Video 一致性，不评估物理合理性、视频质量、风险严重程度、异常时间区间等其他维度。

一、risk_only 和 abnormal 的定义

请严格区分 risk_only 和 abnormal。

risk_only 表示：

- 视频中出现了风险源、危险物体、危险环境或潜在风险条件；
- 但是没有出现明确的人体伤害、摔倒、撞击、烫伤、触电、被攻击、被砸中等直接伤害结果；
- 例如：
  - 儿童靠近热水壶，但没有被烫伤；
  - 老人走在湿滑地面上，但没有摔倒；
  - 野生动物进入门廊或室内，但尚未攻击人；
  - 电线裸露或插座冒火花，但没有人触电；
  - 刀具放在儿童可触及位置，但儿童尚未受伤。

abnormal 表示：

- 视频中不仅出现了风险源或危险条件；
- 还出现了对人的伤害行为、事故结果或直接危险后果；
- 例如：
  - 儿童被玩具绊倒；
  - 老人在浴室滑倒；
  - 人被掉落物砸中；
  - 人被热水烫到；
  - 人触电；
  - 宠物或野生动物攻击人；
  - 车辆失控并对人造成危险或碰撞；
  - 入侵者实施盗窃、破坏或攻击行为。

注意：

- risk_only 不能因为“有风险”就自动当作 abnormal。
- abnormal 不能只出现风险源，必须有明确伤害行为或事故后果。
- 如果 prompt/video_type 是 risk_only，但视频中已经发生了明显伤害，可以在风险类型观察字段中记录，但不要仅因为这一点直接 reject。
- 如果 prompt/video_type 是 abnormal，但视频中只出现风险源、没有伤害行为，也可以在风险类型观察字段中记录，但不要仅因为这一点直接 reject。
- normal 视频不需要调用视觉大模型进行描述，也不需要进行一致性评分，直接跳过。

二、视频描述步骤

对于 video_type 为 risk_only 或 abnormal 的视频，请先调用视觉大模型生成客观视频描述。

视频描述 prompt 使用：

“Please objectively describe the video content. Focus on the scene, main subjects, key objects, risk sources, actions, and whether any harm or abnormal event occurs. Do not infer invisible information. Keep the description concise but specific.”

中文版本可以是：

“请客观描述该视频内容，重点关注场景、主要主体、关键物体、风险源、动作，以及是否发生了对人的伤害行为或异常事件。不要推断画面中不可见的信息。描述应简洁但具体。”

生成的视频描述保存为：

video_description

三、Prompt-Video 一致性评估

请把原始 prompt、video_description 和 expected video_type 一起输入给大模型，让模型判断视频是否与 prompt 语义一致。

评估 prompt 使用：

“Given the original prompt, the expected video type, and the video description, evaluate whether the video content is semantically consistent with the prompt. The wording does not need to be identical. Focus on whether the scene, subject, risk source/key objects, event type, and expected video type are consistent.

The expected video type can be:
- risk_only: risk source or risky condition is visible, but no direct harm to a person occurs.
- abnormal: risk source is visible and direct harm, accident, or dangerous consequence to a person occurs.

Return a JSON object with scores and a short explanation.”

四、评估指标

请输出以下指标：

1. scene_match：场景是否一致，取值 0/1/2。

- 2：场景明确一致；
- 1：场景大体接近，但有轻微偏差或不清楚；
- 0：场景明显不一致。

2. subject_match：主体是否一致，取值 0/1/2。

- 2：主体明确一致；
- 1：主体大体一致，但年龄、身份或类别不够清楚；
- 0：主体明显不一致或缺失。

3. risk_source_match：风险源或关键物体是否一致，取值 0/1/2。

- 2：prompt 中的关键风险源或物体清楚出现；
- 1：出现了相近风险源或物体，但不完全一致或不够清楚；
- 0：关键风险源或物体缺失。

4. event_type_match：风险/异常事件类型是否一致，取值 0/1/2。

- 2：事件类型明确一致；
- 1：事件大体接近，但动作或结果不够完整；
- 0：事件类型明显不一致或关键事件没有发生。

5. video_type_observation：视频风险类型观察结果，取值 0/1/2，仅用于展示，不参与总分，也不作为 final_decision 的关键门控条件。

对于 expected video_type = risk_only：

- 2：视频中出现风险源，但没有明确伤害行为，符合 risk_only 的状态；
- 1：风险源存在，但是否造成伤害不够清楚；
- 0：没有风险源，或已经发生明显伤害，导致其不符合 risk_only 的状态。

对于 expected video_type = abnormal：

- 2：视频中出现风险源，并且发生明确伤害行为或事故后果，符合 abnormal 的状态；
- 1：风险源存在，但伤害行为不够清楚或结果较弱；
- 0：没有风险源，或只有风险源但没有伤害行为，导致其不符合 abnormal 的状态。

注意：

- video_type_observation 只用于展示视频更接近 normal、risk_only 还是 abnormal。
- video_type_observation 不参与总分计算。
- video_type_observation 不作为自动接受或拒绝的硬性条件。
- 即使 video_type_observation = 0，也不要仅因为这一项直接 reject，需要结合 event_type_match、semantic_description_score 和总分判断。

6. semantic_description_score：整体语义一致性分数，取值范围为 [0, 1]。

评分标准：

- 0.90–1.00：视频描述与 prompt 高度一致，场景、主体、风险源、事件类型和视频类型基本都匹配。
- 0.70–0.89：整体一致，但存在轻微缺失或细节偏差，不影响核心事件判断。
- 0.50–0.69：部分一致，核心内容接近，但存在明显缺失或偏差。
- 0.30–0.49：弱一致，只在少数元素上相似，核心事件或视频类型不够匹配。
- 0.00–0.29：基本不一致。

五、总分计算

请计算：

prompt_video_consistency_score =
scene_match
+ subject_match
+ risk_source_match
+ event_type_match
+ 2 * semantic_description_score

总分范围为 0–10 分。

注意：

- video_type_observation 不直接加入总分。
- video_type_observation 仅作为辅助展示字段，用于提示视频更接近 normal、risk_only 还是 abnormal，不作为最终判定的关键门控条件。
- final_decision 主要根据 prompt_video_consistency_score、event_type_match 和 semantic_description_score 判断。
- semantic_description_score 必须限制在 [0, 1]。
- prompt_video_consistency_score 保留两位小数。

六、最终判定 final_decision

对于 risk_only 和 abnormal：

accept：

- prompt_video_consistency_score >= 8；
- event_type_match >= 1；
- semantic_description_score >= 0.7。

weak_accept：

- prompt_video_consistency_score 在 5–7.99 之间；
- 或者总分 >= 8，但 event_type_match 或 semantic_description_score 不够明确；
- 或 observed_video_type = unclear，但核心内容大体接近。

reject：

- prompt_video_consistency_score < 5；
- 或 event_type_match = 0；
- 或 semantic_description_score < 0.5。

特别注意：

- video_type_observation 只用于展示和后续人工分析，不作为自动拒绝或自动接受的硬性条件。
- 即使 video_type_observation = 0，也不要仅因为这一项直接 reject。
- 对于 risk_only 视频，如果视频中出现了风险源，并且和 prompt 的核心风险事件基本一致，即使模型认为 observed_video_type 更接近 abnormal，也只在 video_type_observation 中记录，不直接作为 reject 条件。
- 对于 abnormal 视频，如果视频中出现了 prompt 描述的核心异常事件，但伤害后果不够清楚，也不要仅因为 video_type_observation 不满分而直接 reject，可以根据 event_type_match 和 semantic_description_score 判为 weak_accept。
- 不要因为 risk_only 没有人受伤就把它判为失败；risk_only 的目标本来就是“有风险源但尚未伤害人”。
- 不要因为 video_type_observation 与 expected video_type 不完全一致就直接否定样本；该字段主要用于展示风险类型判断结果。
- 对于野生动物出现在家庭环境中的 risk_only 视频，只要野生动物确实出现，且 prompt 也描述了类似动物出现、动物入侵或动物导致潜在风险的事件，应认为核心事件类型基本一致。
- 对于野生动物攻击人、撞倒人、追逐人并造成危险后果的 abnormal 视频，如果 prompt 也描述了类似动物攻击或动物导致伤害事件，应认为核心事件类型一致。
```