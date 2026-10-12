# 本次整理实际执行的检查

以下检查在新发布文件夹中执行，使用原工程的Python3.9环境，不对原工程安装、重拟合或重绘。命令中的`../Invariant-EvAM/.venv/bin/python`是本次实际解释器；另一个机器安装依赖后可用自己的`python`替换。

```bash
PYTHONDONTWRITEBYTECODE=1 ../Invariant-EvAM/.venv/bin/python -B scripts/verify_release.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../Invariant-EvAM/.venv/bin/python -B -m pytest -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 ../Invariant-EvAM/.venv/bin/python -B scripts/check_g2_backend.py
git status --short
```

正式清单核验：PASS，59张单图、118个PDF/SVG，作图数据、嵌套引用及SOURCE_MAP哈希无错误。

核心测试：15 passed。检查覆盖最小模型、CTMC公共API和投影信息计算；打包检查没有重新执行整套G1–G5研究实验。

G2冻结后端检查：PASS，概率和为0.9999999999999999，解析Jacobian与中心差分的最大绝对误差3.045060731343696e-12。此检查在原Mach-O x86_64后端可运行的平台上执行，不代表在Linux或ARM上重建该库。

Git在本地初始化main分支，文件尚未提交；无远端、无推送。大型原始下载、环境、vendor与截图外置，正式PDF/SVG、实验结果和原代码逐文件保留来源哈希。

额外独立核验脚本及完整源目录比较记录保存在包外构建目录。该比较需要原研究工程；公开可用的只读核验器是本包`scripts/verify_release.py`。
