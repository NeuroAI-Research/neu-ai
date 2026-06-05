from graphviz import Digraph


def get_name(x):
    return f"{x.__name__}\n{getattr(x, "zh", "")}"


def plot_animal(Ani, save_name):
    g = Digraph(
        graph_attr={"rankdir": "LR", "label": get_name(Ani)},
        node_attr={"shape": "plain"},
    )
    colors = {"sensors": "red", "nns": "blue", "outputs": "gray"}
    for key, color in colors.items():
        for x in getattr(Ani, key, []):
            name, label = x.__name__, get_name(x)

            if hasattr(x, "components"):
                with g.subgraph(name=f"cluster_{name}") as sg:
                    sg.attr(label=label)
                    for c in x.components:
                        sg.node(c.__name__, get_name(c))
                continue

            g.node(name, label, color=color, shape="ellipse")
            inputs = getattr(x, "inputs", [])
            outputs = getattr(x, "outputs", [])
            if isinstance(inputs, list):
                for y in inputs:
                    g.edge(getattr(y, "__name__", y), name)
            if isinstance(outputs, list):
                for y in outputs:
                    g.edge(name, getattr(y, "__name__", y))

    g.render(save_name, format="svg", cleanup=True)


# ========================


class Sensor:
    pass


class NN:
    pass


class Muscles:
    zh = "肌肉"


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
    zh = "光感受器"
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


class SupraesophagealGanglion(NN):
    zh = "食道上神经节"
    ancestor = CerebralGanglia
    inputs = [Photoreceptors, Mechanoreceptors, Chemoreceptors]
    function = (
        "最高级整合中心; 具有抑制性控制(切除后虫体过度活跃); 协调复杂行为序列如掘土"
    )


class SubesophagealGanglion(NN):
    zh = "食道下神经节"
    ancestor = None
    inputs = [SupraesophagealGanglion, "口腔感觉"]
    function = "口部运动控制(摄食); 脑与躯干神经链之间的信号中继和调制"


class VentralNerveChain(NN):
    zh = "腹神经链"
    ancestor = VentralNerveCord
    inputs = [SubesophagealGanglion, "本体节感觉"]
    outputs = [Muscles]
    function = "每节半自主的运动控制; 协调蠕动波(peristalsis): 相邻节段顺序收缩/舒张; 即使与脑断开, 各节段仍可产生局部反射"


class GiantNerveFibers(NN):
    zh = "巨型神经纤维"
    ancestor = None
    inputs = [Mechanoreceptors]
    outputs = [Muscles]
    function = "高速逃跑反射: 绕过逐节传导延迟, 毫秒级触发全身同步收缩(缩回地穴); 是专用快速通道电路的原型"


# ------------------------


class Earthworm(Animal):
    zh = "蚯蚓"
    sensors = [Photoreceptors, Mechanoreceptors, Chemoreceptors]
    nns = [
        SupraesophagealGanglion,
        SubesophagealGanglion,
        VentralNerveChain,
        GiantNerveFibers,
    ]
    outputs = [Muscles]


# ============================


class OpticLobeLamina(NN):
    zh = "视叶-板层"
    ancestor = Photoreceptors
    inputs = [Photoreceptors]
    function = "亮度对比度增强(侧抑制); 运动方向检测的第一步; 颜色初步分离"


class OpticLobeMedulla(NN):
    zh = "视叶-髓质"
    ancestor = Photoreceptors
    inputs = [OpticLobeLamina, Photoreceptors]
    function = (
        "时空视觉特征提取; ON/OFF通道分离(亮度增加/减少); 颜色处理; 运动检测的核心计算"
    )


class OpticLobeLobula(NN):
    zh = "视叶-小叶"
    ancestor = Photoreceptors
    inputs = [OpticLobeMedulla]
    function = "复杂图案识别(天敌轮廓, 花纹); 光流方向性检测, 支持飞行稳定和降落"


# -------------------------


class AntennalLobe(NN):
    zh = "触角叶"
    ancestor = "原始化学感受中枢"
    inputs = [Chemoreceptors]
    function = "嗅觉处理(功能类似脊椎动物嗅球); 肾小球(glomeruli)结构实现气味分类; 侧抑制实现气味对比度增强"


# -------------------------


class MushroomBodyCalyx(NN):
    zh = "蘑菇体萼"
    ancestor = "原始嗅觉中枢神经节"
    inputs = [AntennalLobe, OpticLobeLobula]
    function = "稀疏编码: 将密集感觉信号转换为稀疏的Kenyon细胞激活模式, 提高模式分离能力, 是记忆存储的基础"


class MushroomBodyLobes(NN):
    zh = "蘑菇体叶"
    ancestor = "原始嗅觉中枢神经节"
    inputs = [MushroomBodyCalyx]
    function = "记忆读出: 奖励多巴胺信号修改Kenyon细胞->输出神经元突触权重; γ叶->短期记忆, α/β叶->长期记忆"


class MushroomBodies(NN):
    zh = "蘑菇体"
    ancestor = "原始嗅觉中枢神经节"
    function = "联合学习与记忆中枢; 实现条件反射(气味+奖惩->行为); 多模态感觉整合; 在蜜蜂中支持导航记忆和认知图谱"
    components = [MushroomBodyCalyx, MushroomBodyLobes]


# ---------------------------


class EllipsoidBody(NN):
    zh = "椭圆体"
    ancestor = None
    inputs = [OpticLobeLobula]
    function = "头部朝向的环形编码(ring attractor network); 维持稳定的方向感知, 即使视觉输入短暂消失也能保持"


class FanShapedBody(NN):
    zh = "扇形体"
    ancestor = None
    inputs = [MushroomBodyLobes, EllipsoidBody]
    function = "将当前朝向信息与目标方向对比, 计算转向量; 支持睡眠调节(果蝇研究发现)"


class CentralComplex(NN):
    zh = "中央复合体"
    ancestor = None
    function = "导航与空间定向中枢; 计算头部朝向角度(类似罗盘); 整合路径积分信息; 产生定向行走的运动指令"
    components = [EllipsoidBody, FanShapedBody]


# -------------------------


class SubesophagealGanglion2(NN):
    zh = "食道下神经节2"
    ancestor = SubesophagealGanglion
    inputs = ["味觉(口器)", "触觉(口部)", AntennalLobe]
    outputs = [Muscles]
    function = "进食决策(饥饿+食物检测->摄食); 口部运动控制; 味觉处理"


class ThoracicGanglia(NN):
    zh = "胸部神经节"
    ancestor = VentralNerveChain
    inputs = [FanShapedBody, SubesophagealGanglion2, Mechanoreceptors]
    outputs = [Muscles]
    function = "行走和飞行的中央模式发生器(CPG); 可在脑断开后自主产生节律运动; 三对腿的协调步态计算"


class AbdominalGanglia(NN):
    zh = "腹部神经节"
    ancestor = VentralNerveChain
    inputs = ["腹部感觉", "生殖器感觉"]
    outputs = [Muscles, "生殖系统"]
    function = "腹部运动控制; 产卵行为的局部控制"


# ---------------------------


class Insect(Animal):
    zh = "昆虫"
    sensors = [Photoreceptors, Chemoreceptors, Mechanoreceptors]
    nns = [
        OpticLobeLamina,
        OpticLobeMedulla,
        OpticLobeLobula,
        AntennalLobe,
        MushroomBodyCalyx,
        MushroomBodyLobes,
        MushroomBodies,
        EllipsoidBody,
        FanShapedBody,
        CentralComplex,
        SubesophagealGanglion2,
        ThoracicGanglia,
        AbdominalGanglia,
    ]
    outputs = [Muscles]


if __name__ == "__main__":
    plot_animal(Insect, "temp")
