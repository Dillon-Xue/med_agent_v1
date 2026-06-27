import os, logging
from dotenv import load_dotenv
from tools.approval_tool import ApprovalTool
load_dotenv()
from tools.file_tool import FileTool

logger = logging.getLogger(__name__)
BASE_DIR = os.getenv("MED_AGENT_ROOT", os.getcwd())

TOOLS = {}

def get_api_key():
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("❌ DASHSCOPE_API_KEY 未加载成功")
    return key

def init_tools():
    global TOOLS
    api_key = get_api_key()

    from tools.rag_tool import RAGTool
    from tools.drug_tool import DrugTool
    from tools.literature_tool import LiteratureTool
    from tools.guideline_tool import GuidelineTool
    from tools.risk_tool import RiskTool
    from tools.patient_tool import PatientTool   # 导入 MySQL 版
    from tools.report_tool import ReportTool

    TOOLS = {
        "rag": RAGTool(BASE_DIR, api_key),
        "drug": DrugTool(BASE_DIR, api_key),
        "literature": LiteratureTool(BASE_DIR, api_key),
        "guideline": GuidelineTool(BASE_DIR, api_key),
        "risk": RiskTool(BASE_DIR, api_key),
        "patient": PatientTool(),   # 无需额外参数，从环境变量读取配置
        "report": ReportTool(),
        "approval": ApprovalTool(),
        "file": FileTool(api_key),
    }

    logger.debug("\n========== TOOL INIT OK ==========")
    for k in TOOLS:
        logger.info(f"[tool loaded] {k}")

    return TOOLS

def get_tools():
    global TOOLS
    if not TOOLS:
        return init_tools()
    return TOOLS

def get_tool_trace(intent: str):
    tool = TOOLS.get(intent)
    if not tool:
        return []
    if hasattr(tool, "get_trace"):
        return tool.get_trace()
    return []