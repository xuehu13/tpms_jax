# tpms_jax

三维 TPMS 隐式几何 → HEX8 体素有限元(JAX-FEM)→ 自动微分 → 结构逆设计的科研项目。

## 环境

使用 [Pixi](https://pixi.prefix.dev) 管理,环境配置见 `pixi.toml`,版本锁定见 `pixi.lock`。
当前 Pixi 环境安装 PyPI `jax-fem==0.0.12`；独立的 JAX-FEM 源码目录仅供参考。
实际安装版本、关键源码指纹及 M4 验证记录见
[`validation/m4_review/`](validation/m4_review/)。

```bash
pixi install        # 按 pixi.lock 创建/恢复环境
pixi run test       # 运行环境自检与有限元 smoke test
pixi run python     # 进入项目 Python 环境
```

## 状态

- M0:环境初始化与基础验证(2026-09)
- 里程碑规划:M1 TPMS 隐式几何与连续密度场 → M2 体素有限元 → M3 自动微分与逆设计
