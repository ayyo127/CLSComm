from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

# 定义日志文件路径
log_path = "/home/lsy224826/project/CommFormer-orig/commformer/scripts/results/StarCraft2/3m/commformer_dec/single/CommFormer-241111-155351/logs/eval_win_rate/eval_win_rate/events.out.tfevents.1731311664.user-PowerEdge-T640"

# 初始化 EventAccumulator
event_acc = EventAccumulator(log_path)
event_acc.Reload()  # 加载数据


# 获取指定标量数据
scalar_tags = event_acc.Tags()['scalars']  # 获取所有标量标签
for tag in scalar_tags:
    # 提取数据
    events = event_acc.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    wall_times = [e.wall_time for e in events]
    print("成功")
    # 转为 DataFrame
    df = pd.DataFrame({
        "wall_time": wall_times,
        "step": steps,
        "value": values
    })
    print("成功2")

    # 保存为 CSV
    df.to_csv(f"3m-commfommer-{tag}.csv", index=False)
    print(f"导出 {tag}.csv 完成！")
