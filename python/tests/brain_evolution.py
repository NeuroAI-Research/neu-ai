from graphviz import Digraph


def get_name(x):
    return f"{x.__name__}\n{getattr(x, "zh", "")}"


def plot_animal(Ani, save_name):
    g = Digraph(graph_attr={"rankdir": "LR", "label": get_name(Ani)})
    colors = {"sensors": "red", "nns": "blue", "outputs": "gray"}
    for key, color in colors.items():
        for x in getattr(Ani, key, []):
            name = x.__name__
            g.node(name, get_name(x), color=color)
            inputs = getattr(x, "inputs", [])
            outputs = getattr(x, "outputs", [])
            if isinstance(inputs, list):
                for y in inputs:
                    g.edge(y.__name__, name)
            if isinstance(outputs, list):
                for y in outputs:
                    g.edge(name, y.__name__)
    g.render(save_name, format="svg", cleanup=True)


# ========================


class Sensor:
    pass


class NN:
    pass


class Muscles:
    pass


class Animal:
    pass


# ==============================


class Mechanoreceptors(Sensor):
    zh = "机械感受器"
    ancestor = None
    inputs = "水流振动, 接触压力"
    function = "触觉和水流检测, 用于逃避捕食者"


class Statocysts(Sensor):
    zh = "平衡囊"
    ancestor = None
    inputs = "重力方向, 加速度"
    function = "重力方向感知, 实现身体在水中的定向; 是最早的前庭感觉器官原型"


class Photoreceptors(Sensor):
    zh = "感光细胞"
    ancestor = None
    inputs = "光强度变化"
    function = "光强度检测 (不能辨别方向); 利用光变化触发逃避行为"


# ------------------------------


class DiffuseNerveNet(NN):
    zh = "弥散神经网"
    ancestor = "由化学信号细胞进化为真正神经元, 形成网状连接"
    inputs = [Mechanoreceptors, Statocysts, Photoreceptors]
    outputs = [Muscles]
    function = "分布式感觉-运动整合; 无中央处理中心, 刺激可双向传导; 实现基础逃避反射和捕食反射"


# ----------------------------------


class Jellyfish(Animal):
    zh = "水母"
    sensors = [Mechanoreceptors, Statocysts, Photoreceptors]
    nns = [DiffuseNerveNet]
    outputs = [Muscles]


# ====================================


class Eyecups(Sensor):
    zh = "眼杯"
    ancestor = Photoreceptors
    inputs = "具有方向性的光线 (色素杯阻挡一侧光)"
    function = "方向性光感知 (能辨别光从哪个方向来); 用于趋光/避光行为导航; 是视觉方向计算的起点"


class Chemoreceptors(Sensor):
    zh = "化学感受器"
    ancestor = None
    inputs = "水中化学梯度 (食物气味, 危险化学物质)"
    function = "定向嗅觉: 比较两侧化学浓度差, 计算梯度方向, 引导趋食行为"


# -----------------------------


class CerebralGanglia(NN):
    zh = "脑神经节"
    ancestor = DiffuseNerveNet
    inputs = [Eyecups, Chemoreceptors, Mechanoreceptors]
    function = "第一个中央化信息处理中心; 实现分层控制(脑->神经索->肌肉); 头化(cephalization) 的起点"


class VentralNerveCord(NN):
    zh = "腹神经索"
    ancestor = DiffuseNerveNet
    inputs = [CerebralGanglia]
    outputs = [Muscles]
    function = (
        "纵向信息高速公路; 实现从头到尾的协调运动; 比弥散神经网更快, 更定向的信号传导"
    )


# --------------------------


class Planaria(Animal):
    zh = "涡虫"
    sensors = [Eyecups, Chemoreceptors, Mechanoreceptors]
    nns = [CerebralGanglia, VentralNerveCord]
    outputs = [Muscles]


# =============================


class Thermoreceptors(Sensor):
    zh = "热感受器"
    ancestor = None
    inputs = "温度变化"
    function = (
        "温度梯度检测与记忆: 记住`培养温度`并驱使虫体回到该温度, 是最早的温度记忆计算"
    )


# ----------------------------


class PharyngealNerveRing(NN):
    zh = "咽神经环"
    ancestor = CerebralGanglia
    inputs = [Mechanoreceptors, Chemoreceptors]
    outputs = [Muscles]
    function = "进食控制回路; 检测食物存在并协调咽部泵吸节律, 与主神经环半独立运行"


class CommandInterneurons(NN):
    zh = "命令中间神经元"
    ancestor = None
    inputs = [Mechanoreceptors, Chemoreceptors, Thermoreceptors]
    function = "多模态感觉整合->行为选择; 是302个神经元中的`决策层`"


class MotorInterneurons(NN):
    zh = "运动中间神经元"
    ancestor = None
    inputs = [CommandInterneurons, Mechanoreceptors]
    function = "前进/后退决策回路"


class MotorNeurons(NN):
    zh = "运动神经元"
    ancestor = VentralNerveCord
    inputs = [MotorInterneurons]
    outputs = [Muscles]
    function = "蛇形爬行的中央模式发生器(CPG): 背腹肌肉交替收缩产生正弦波运动"


# --------------------------


class Nematode(Animal):
    zh = "线虫"
    sensors = [Mechanoreceptors, Chemoreceptors, Thermoreceptors]
    nns = [PharyngealNerveRing, CommandInterneurons, MotorInterneurons, MotorNeurons]
    outputs = [Muscles]


# ============================

if __name__ == "__main__":
    plot_animal(Nematode, "temp")
