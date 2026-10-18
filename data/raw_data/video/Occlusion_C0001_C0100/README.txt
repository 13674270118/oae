遮挡外观一致性候选视频 / Batch 1
编号范围：C0001–C0100
本包包含 100 个模型生成MP4，未重新编码，保留原始画质。
videos/ 内文件名与之前Excel清单的Candidate ID对应。
manifest.json 包含原始任务、场景指令、模型名、输入片段与参考视频链接及人工复核字段。
候选按任务与指令初筛，并非已确认全部合格或全是外观错误。
请检查：完全遮挡、目标重新出现、前后可比较，再判断外观是否一致。
部分MP4只包含生成后续。需要时从清单或manifest中的input_video_url观看前半段。
相同group_id的样本应留在同一数据划分中。
来源：https://huggingface.co/datasets/Hokin/object-permanence-benchmark
许可：CC BY-NC 4.0；非商业研究使用需署名并引用 https://arxiv.org/abs/2609.28654
