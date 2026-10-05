# 原生Abaqus同材料场、指定变形桥接

point_check为单个非均匀占据单元的20%仿射检查；saved20为原HEX8 N64全部保存位移被指定的状态检查。SDVINI冻结参考Gauss占据，UHYPER使用相同刚度加权Neo-Hookean；没有USDFLD/VUSDFLD逐增量重填占据。原生C3D8积分处理仍与JAX一般HEX8不同。

单元解析反力/能量差约6.46e-9/2.63e-8；完整保存状态反力/能量差1.13%/1.08%。所有位移被指定，**不是独立0至20%平衡路径**，单步ALLWK也不作为路径功认证。

input/expected/result/mapping_audit记录数据身份和核对；execution.json说明实际编译接口。C++文件只实现SDVINI/UHYPER，单元计算仍由Abaqus原生负责。配置依赖本机VS、原安装DLL的未修改副本和E盘作业目录；不是跨机器完整可复现打包。大INP、Gauss数据、DLL及ODB留在E盘原作业处，不改安装目录。链接排障不作力学失败。
