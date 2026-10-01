# 会话交接（Agent 框架决定）

回测引擎继续写在本仓库。Agent 以后放在同仓库的 `agent/`，只调用 `Backtest.run` 的后继接口，不拆成两个 Git 仓库。

## 引擎已经锁死的部分
成交时点、手续费、滑点、现金与持仓结算、绩效公式。Agent 不能改这些。

## Agent 以后可以提交的 Spec
hypothesis、信号类型（先只允许 momentum/MACD）、窗口参数、允许交易的 volume/volatility regime、是否做波动率缩放、目标波动率和最大杠杆。

## 框架
- Orchestrator：普通 Python 循环，负责停止和选择当前最优。
- Researcher：大模型，只输出 StrategySpec。
- Reviewer：回测前审查，打回的 Spec 不回测。
- 数据分三段：搜索段给 Researcher 反复回测；选择段只给 Orchestrator 挑 Spec；测试段在框架定稿后跑一次。

## 对比
A 手写基线，B 只有 Researcher，C Researcher + Reviewer，D 两个 Researcher 并行后取选择段更好的一份。一次只改一个旋钮：反馈内容、记忆长度、选优规则、停止条件、Reviewer 标准。每个框架 5 个种子，看选择段与测试段的差距。

## 写法
教练式，一步一步写。下一节是 Session 5：在现有 T+1 开盘成交上加入手续费和滑点。