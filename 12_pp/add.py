'''
两个张量的相加
'''
import paddle.fluid as fluid

# 定义
x = fluid.layers.fill_constant(shape=(1,),
                               dtype='float32',
                               value=5.0)
y = fluid.layers.fill_constant(shape=(1,),
                               dtype='float32',
                               value=(100.0))
res = x + y

# 执行
place = fluid.CPUPlace()
exe = fluid.Executor(place=place)

result = exe.run(program=fluid.default_main_program(),
                 fetch_list=[res])  # 获取哪个op的结果

print(result[0][0])  # [结果]   [[105.0]]
