# 项目背景与研究上下文 (Project Context & Mission)

## 1. 核心目标
本工程目标是：严格跟着课程大纲（Project_proposal_version_4.pdf 的 Session 1-10），从零开始手搓一个纯原生、透明、严格无前视偏差（No Lookahead Bias）的面向对象（OOP）CTA 策略回测平台。

## 2. 核心学术与前沿理论依托
1. **Jump Trading (ICML 2026 Expo)**:
   - 核心原则：极度克制的最小可度量闭环（Minimal Measurable Loop）。
   - 时序因果铁律（Strict Point-in-Time）：严防前视偏差；必须包含真实摩擦（交易成本、滑点、严格的 T+1 撮合延时）。
2. **Google Research / MIT (arXiv:2512.08296)**:
   - 智能体系统规律：可解耦的金融任务中，中心化校验（Centralized Verification）能将错误级联放大率从 17.2x 压制到 4.4x。
   - 本项目后续将作为确定性工具层，支持 LLM Agent 进行策略假说生成与参数搜索。

## 3. 课程 10 个 Session 任务清单
- Session 1: 研究设计（从交易想法到可检验的基线）。
- Session 2: 手搓透明 Class-based 回测器（6 大账本：现金、持仓、市值、权益、盈亏、手续费）。
- Session 3-4: 朴素动量（Momentum）与 MACD 信号生成与统计诊断（夏普、回撤、偏度、换手率）。
- Session 5: 注入逼真摩擦（手续费、滑点、T+1 撮合时滞）。
- Session 6-7: 滚动成交量与波动率的 4 象限市场状态（Regime）划分与条件收益诊断。
- Session 8: 波动率自适应缩放（Volatility Scaling）与策略修正。
- Session 9-10: 机器学习增量评估（Ridge/Lasso/GBDT，检验是否真有 Out-of-Sample 泛化价值）。

## 4. 协作模式规定
- **绝不大段直接贴出全部写好的大文件代码**。
- **教练式手搓模式**：一步一步带我分析逻辑、让我自己写或带着我敲，深入理解每一个时序推进细节与金融账本结算机制。