from dataclasses import dataclass, field
from typing import List
import topic

@dataclass
class SlaveInfo:
    slave_name: str = ""       # 从站名称
    phy_id: int = 0            # 物理地址
    alias: int = 0             # 逻辑别名
    slave_state: int = 0       # 从站状态机
    is_online: bool = False    # 是否在线
    is_virtual: bool = False   # 是否为虚拟/仿真从站
    is_error: bool = False     # 是否存在错误

@dataclass
class FtvalueInfo:
    fx: float = 0.0  # X方向力
    fy: float = 0.0  # Y方向力
    fz: float = 0.0  # Z方向力
    mx: float = 0.0  # X方向力矩
    my: float = 0.0  # Y方向力矩
    mz: float = 0.0  # Z方向力矩

@dataclass
class ControllerInfo:
    controller_name: str = ""  # 控制器名称
    control_cycle: float = 0.0 # 控制周期
    global_count: int = 0      # 全局计数
    master_info: str = ""      # 主控信息
    is_link_up: bool = False   # 与机器人的链路是否在线
    ftvalues: List[FtvalueInfo] = field(default_factory=list)  # 六维力传感器数据

@dataclass
class ModelsInfo:
    model_name: str = ""       # 模型名称
    model_type: str = ""       # 模型类型
    is_using_sp: bool = False
    is_collision_detection: bool = False

@dataclass
class ModelInfo:
    error_code: int = 0        # 错误码
    error_msg: str = ""        # 错误描述
    model_state: int = 0       # 模型状态
    model_time_rate: float = 0.0 # 模型运行时间比例或速率
    current_func_name: str = "" # 当前正在执行的函数名
    current_func_info: str = "" # 当前函数的附加信息
    func_count: int = 0        # 函数计数或已调用次数

@dataclass
class JointInfo:
    joint_type: str = ""       # 关节类型
    position: float = 0.0      # 当前位置
    torque: float = 0.0        # 当前力矩
    is_enabled: bool = False   # 是否上电使能
    mode: int = 0              # 控制模式
    error_code: int = 0        # 错误码
    digit_output: int = 0      # 数字输出状态
    digit_input: int = 0       # 数字输入状态
    max_position: float = 0.0  # 关节最大位置限制
    min_position: float = 0.0  # 关节最小位置限制
    max_vel: float = 0.0       # 最大速度限制
    min_vel: float = 0.0       # 最小速度限制
    max_acc: float = 0.0       # 最大加速度限制
    min_acc: float = 0.0       # 最小加速度限制
    max_collision_torque: float = 0.0 # 碰撞检测力矩阈值

@dataclass
class CurrentPointInfo:
    point_name: str = ""       # 当前点位的名称
    tool_name: str = ""        # 当前使用的工具名称
    wobj_name: str = ""        # 当前工件坐标系名称
    tool_data: List[float] = field(default_factory=list) # 工具的 6D 数据
    wobj_data: List[float] = field(default_factory=list) # 工件坐标系的 6D 数据
    robottarget: List[float] = field(default_factory=list) # 当前笛卡尔目标位姿
    jointtarget: List[float] = field(default_factory=list) # 当前关节目标角度

@dataclass
class ToolInfo:
    tool_name: str = ""        # 工具名称
    data: List[float] = field(default_factory=list) # 工具数据

@dataclass
class WobjInfo:
    wobj_name: str = ""        # 工件坐标系名称
    data: List[float] = field(default_factory=list) # 工件坐标系数据

@dataclass
class LoadInfo:
    load_name: str = ""        # 负载名称
    data: List[float] = field(default_factory=list) # 负载数据

@dataclass
class PointInfo:
    point_name: str = ""
    tool_name: str = ""
    wobj_name: str = ""
    tool_data: List[float] = field(default_factory=list)
    wobj_data: List[float] = field(default_factory=list)
    robottarget: List[float] = field(default_factory=list)
    jointtarget: List[float] = field(default_factory=list)

@dataclass
class SubsystemInfo:
    subsystem_name: str = ""   # 子系统名称
    id: int = 0                # 子系统编号
    state: int = 0             # 运行状态

@dataclass
class SensorInfo:
    sensor_name: str = ""      # 传感器名称
    id: int = 0                # 传感器编号
    state: int = 0             # 传感器状态

@dataclass
class InterfaceInfo:
    interface_name: str = ""   # 接口名称
    id: int = 0                # 接口编号
    state: int = 0             # 接口状态

@dataclass
class SystemStateData:
    header_timestamp: int = 0  # 消息头和时间戳
    header_frame_id: int = 0   # 帧ID
    system_running_state: int = 0 # 系统运行状态

    controller: ControllerInfo = field(default_factory=ControllerInfo)
    models: List[ModelsInfo] = field(default_factory=list)
    models_info: List[ModelInfo] = field(default_factory=list)
    slaves: List[SlaveInfo] = field(default_factory=list)
    models_joints: List[JointInfo] = field(default_factory=list)
    models_tools: List[ToolInfo] = field(default_factory=list)
    models_wobjs: List[WobjInfo] = field(default_factory=list)
    models_loads: List[LoadInfo] = field(default_factory=list)
    models_teach_points: List[PointInfo] = field(default_factory=list)
    models_current_points: List[CurrentPointInfo] = field(default_factory=list)
    model: ModelInfo = field(default_factory=ModelInfo)
    subsystems: List[SubsystemInfo] = field(default_factory=list)
    sensors: List[SensorInfo] = field(default_factory=list)
    interfaces: List[InterfaceInfo] = field(default_factory=list)


def display_rt(tt: topic.SystemRtState, parm: SystemStateData):
    parm.models_joints = []
    parm.models_current_points = []  # 清空当前点位列表
    parm.models_info = []  # 添加模型信息列表
    parm.header_timestamp = tt.head().timestamp()
    parm.header_frame_id = tt.head().frame_id()
    parm.system_running_state = tt.system_running_state()

    parm.controller.controller_name = tt.controller().controller_name()
    parm.controller.control_cycle = tt.controller().control_cycle()
    parm.controller.global_count = tt.controller().global_count()
    parm.controller.master_info = tt.controller().master_info()
    parm.controller.is_link_up = tt.controller().is_link_up()
    
     # 解析六维力传感器数据
    ftvalues = []
    for ftvalue in tt.controller().ftvalues():
        ftvalues.append(FtvalueInfo(
            fx=ftvalue.fx(),
            fy=ftvalue.fy(),
            fz=ftvalue.fz(),
            mx=ftvalue.mx(),
            my=ftvalue.my(),
            mz=ftvalue.mz()
        ))
    parm.controller.ftvalues = ftvalues

    parm.models = [ModelsInfo(model_name=model.model_name(), model_type=model.model_type()) for model in tt.model()]

    for model in tt.model():
        joints = []
        for joint in model.joint():
            joints.append(JointInfo(
                joint_type=joint.joint_type(),
                position=joint.position(),
                torque=joint.torque(),
                is_enabled=joint.is_enabled(),
                mode=joint.mode(),
                error_code=joint.error_code(),
                digit_output=joint.digit_output(),
                digit_input=joint.digit_input()
            ))
        parm.models_joints.extend(joints)

        current_point = model.current_point()
        tool = current_point.tool()
        wobj = current_point.wobj()

        # 为每个模型创建新的 CurrentPointInfo 并添加到列表中
        current_point_info = CurrentPointInfo(
            point_name=current_point.point_name(),
            tool_name=tool.tool_name(),
            wobj_name=wobj.wobj_name(),
            tool_data=list(tool.data()),
            wobj_data=list(wobj.data()),
            robottarget=list(current_point.robottarget()),
            jointtarget=list(current_point.jointtarget())
        )
        parm.models_current_points.append(current_point_info)

        # 为每个模型创建 ModelInfo 并添加到列表中
        model_info = ModelInfo(
            error_code=model.error_code(),
            error_msg=model.error_msg(),
            model_state=model.model_state(),
            model_time_rate=model.model_time_rate(),
            current_func_name=model.current_func_name(),
            current_func_info=model.current_func_info(),
            func_count=model.func_count()
        )
        parm.models_info.append(model_info)


def display_nrt(tt: topic.SystemNrtState, parm: SystemStateData):
    parm.header_timestamp = tt.head().timestamp()
    parm.header_frame_id = tt.head().frame_id()
    parm.system_running_state = tt.system_running_state()

    parm.slaves = [
        SlaveInfo(
            slave_name=slave.slave_name(),
            phy_id=slave.phy_id(),
            alias=slave.alias(),
            slave_state=slave.slave_state(),
            is_online=slave.is_online(),
            is_virtual=slave.is_virtual(),
            is_error=slave.is_error()
        ) for slave in tt.controller().slave()
    ]
    
    for model in tt.model():
        parm.models.append(ModelsInfo(model_name=model.model_name(), model_type=model.model_type(),is_using_sp=model.is_using_sp(),is_collision_detection=model.is_collision_detection()))

        joints = []
        for joint in model.joint():
            joints.append(JointInfo(
                max_position=joint.max_position(),
                min_position=joint.min_position(),
                max_vel=joint.max_vel(),
                min_vel=joint.min_vel(),
                max_acc=joint.max_acc(),
                min_acc=joint.min_acc(),
                max_collision_torque=joint.max_collision_torque()
            ))
        parm.models_joints.extend(joints)

        tools = []
        for tool in model.tools():
            tools.append(ToolInfo(
                tool_name=tool.tool_name(),
                data=list(tool.data())
            ))
        parm.models_tools.extend(tools)

        wobjs = []
        for wobj in model.wobjs():
            wobjs.append(WobjInfo(
                wobj_name=wobj.wobj_name(),
                data=list(wobj.data())
            ))
        parm.models_wobjs.extend(wobjs)

        loads = []
        for load in model.loads():
            loads.append(LoadInfo(
                load_name=load.load_name(),
                data=list(load.data())
            ))
        parm.models_loads.extend(loads)

        teach_points = []
        for point in model.teach_points():
            tool = point.tool()
            wobj = point.wobj()

            teach_points.append(PointInfo(
                point_name=point.point_name(),
                tool_name=tool.tool_name(),
                wobj_name=wobj.wobj_name(),
                tool_data=list(tool.data()),
                wobj_data=list(wobj.data()),
                robottarget=list(point.robottarget()),
                jointtarget=list(point.jointtarget())
            ))
        parm.models_teach_points.extend(teach_points)

    parm.subsystems = [
        SubsystemInfo(
            subsystem_name=subsystem.subsystem_name(),
            id=subsystem.id(),
            state=subsystem.state()
        ) for subsystem in tt.subsystem()
    ]

    parm.sensors = [
        SensorInfo(
            sensor_name=sensor.sensor_name(),
            id=sensor.id(),
            state=sensor.state()
        ) for sensor in tt.sensor()
    ]

    parm.interfaces = [
        InterfaceInfo(
            interface_name=interface.interface_name(),
            id=interface.id(),
            state=interface.state()
        ) for interface in tt.interface()
    ]


def display_basic(tt: topic.SystemRtState, parm: SystemStateData):
    parm.models_joints = []
    parm.models_current_points = []  # 清空当前点位列表
    parm.models_info = []  # 添加模型信息列表

    parm.models = [ModelsInfo(model_name=model.model_name(), model_type=model.model_type()) for model in tt.model()]

    for model in tt.model():
        joints = []
        for joint in model.joint():
            joints.append(JointInfo(
                position=joint.position(),
                is_enabled=joint.is_enabled()
            ))
        parm.models_joints.extend(joints)

        current_point = model.current_point()

        # 为每个模型创建新的 CurrentPointInfo 并添加到列表中
        current_point_info = CurrentPointInfo(
            robottarget=list(current_point.robottarget()),
            jointtarget=list(current_point.jointtarget())
        )
        parm.models_current_points.append(current_point_info)
