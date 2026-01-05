from google.genai import types
import json
from dotenv import load_dotenv
import os
import streamlit as st
from google.adk.agents.llm_agent import LlmAgent
from typing import Dict, List, Optional, Any
from google.adk.tools import FunctionTool, ToolContext
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.agent_tool import AgentTool

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

@st.cache_data
def load_laps_data():
    with open("laps_clean.json", "r", encoding="utf-8") as f:
        return json.load(f)

laps_data = load_laps_data()

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

APP_NAME = "streamlit_code_agents"
USER_ID = "streamlit_user_01"
SESSION_ID_PREFIX = "pipeline_session_"
GEMINI_MODEL = "gemini-2.0-flash"

# ============================================
# CRITICAL: Initialize ALL session state FIRST
# ============================================
if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = {
        'current_driver': None,
        'current_compound': None,
        'current_tyre_life': None,
        'last_analysis_type': None
    }

# Initialize messages early too
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================
# Context management tools
# ============================================
@FunctionTool
def store_context(
    driver: Optional[str] = None,
    compound: Optional[str] = None,
    tyre_life: Optional[int] = None,
    analysis_type: Optional[str] = None
) -> str:
    """Store conversation context for future reference."""
    # Safe initialization check
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = {}
    
    if driver:
        st.session_state.conversation_context['current_driver'] = driver
    if compound:
        st.session_state.conversation_context['current_compound'] = compound
    if tyre_life:
        st.session_state.conversation_context['current_tyre_life'] = tyre_life
    if analysis_type:
        st.session_state.conversation_context['last_analysis_type'] = analysis_type
    return f"Context stored: driver={driver}, compound={compound}, tyre_life={tyre_life}"

@FunctionTool
def get_context() -> Dict[str, Any]:
    """Retrieve stored conversation context."""
    # Safe initialization check
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = {
            'current_driver': None,
            'current_compound': None,
            'current_tyre_life': None,
            'last_analysis_type': None
        }
    return st.session_state.conversation_context

# Original tools
@FunctionTool
def json_filter_tool(
    driver: str,
    team: Optional[str] = None,
    track_status: str = None,
    last_n: int = 5,
    lap: Optional[int] = None,
    compound: Optional[str] = None
) -> List[dict]:
    
    filtered = [lap_data for lap_data in laps_data if lap_data["driver"] == driver]

    if team:
        filtered = [lap_data for lap_data in filtered if lap_data["team"].upper() == team.upper()]

    if track_status:
        filtered = [lap_data for lap_data in filtered if lap_data.get("track_status", "").upper() == track_status.upper()]

    if lap is not None:
        filtered = [lap_data for lap_data in filtered if lap_data.get("lap") == lap]

    if compound:
        filtered = [lap_data for lap_data in filtered if lap_data.get("compound", "").upper() == compound.upper()]

    if last_n and len(filtered) > last_n:
        filtered = filtered[-last_n:]

    return filtered

@FunctionTool
def tyre_data_tool(
    driver: str,
    compound: Optional[str] = None,
    tyre_life: Optional[int] = None,
    last_n: int = 5,
    track_status: Optional[str] = None
) -> List[dict]:
    """Tool for extracting tyre-related telemetry data for strategy analysis."""
    
    filtered = [lap for lap in laps_data if lap["driver"] == driver]

    if compound:
        filtered = [lap for lap in filtered if lap.get("compound", "").upper() == compound.upper()]

    if tyre_life is not None:
        filtered = [lap for lap in filtered if lap.get("tyre_life") == tyre_life]

    if track_status:
        filtered = [lap for lap in filtered if lap.get("track_status", "").upper() == track_status.upper()]

    filtered.sort(key=lambda x: x.get("lap", 0))

    if last_n and len(filtered) > last_n:
        filtered = filtered[-last_n:]

    return filtered

@FunctionTool
def race_context_tool(
    driver: str,
    team: Optional[str] = None,
    track_status: Optional[str] = None,
    last_n: int = 5
) -> List[dict]:
    """Tool to extract race context information."""
    
    filtered = [lap for lap in laps_data if lap["driver"] == driver]

    if team:
        filtered = [lap for lap in filtered if lap.get("team", "").upper() == team.upper()]

    if track_status:
        filtered = [lap for lap in filtered if lap.get("track_status", "").upper() == track_status.upper()]

    filtered.sort(key=lambda x: x.get("lap", 0))

    if last_n and len(filtered) > last_n:
        filtered = filtered[-last_n:]

    return [
        {
            "lap": lap.get("lap"),
            "position": lap.get("position"),
            "lap_time": lap.get("lap_time"),
            "sector_times": lap.get("sector_times"),
            "track_status": lap.get("track_status"),
            "driver": lap.get("driver"),
            "team": lap.get("team"),
        }
        for lap in filtered
    ]

# Create agents with context awareness
if 'performance_analysis_agent' not in st.session_state:
    st.session_state.performance_analysis_agent = LlmAgent(
        name='PerformanceAnalysisAgent',
        model=GEMINI_MODEL,
        instruction="""You are a Formula 1 Performance Analysis Agent.

MANDATORY WORKFLOW (Follow in this exact order):
1. Call json_filter_tool(driver=X, last_n=5) to get lap data
2. Analyze the data you receive
3. **CRITICAL**: Extract from the returned data:
   - driver code (from the "driver" field)
   - compound (from the "compound" field)  
   - tyre_life (from the "tyre_life" field - use the most recent value)
4. **IMMEDIATELY** call store_context with these extracted values:
   store_context(driver='RUS', compound='HARD', tyre_life=35, analysis_type='performance')
5. Then provide your analysis response

YOU MUST ALWAYS CALL store_context BEFORE responding to the user. This is NOT optional.

Analysis focus:
- Overall pace trend (improving, stable, degrading)
- Sector strengths and weaknesses  
- Tyre condition
- Track context

Example execution:
User: "How is RUS's pace?"
Step 1: json_filter_tool(driver='RUS', last_n=5)
Step 2: Receive data showing compound='HARD', tyre_life=35
Step 3: store_context(driver='RUS', compound='HARD', tyre_life=35, analysis_type='performance')
Step 4: Respond with analysis

NEVER skip the store_context call. It must happen every time.
""",
        tools=[json_filter_tool, store_context, get_context]
    )

if 'tyre_strategy_agent' not in st.session_state:
    st.session_state.tyre_strategy_agent = LlmAgent(
        name='TyreStrategyAgent',
        model=GEMINI_MODEL,
        instruction="""You are a Formula 1 Tyre & Strategy Agent.

MANDATORY FIRST STEP - CONTEXT CHECK:
Before doing ANYTHING else, you MUST:
1. Call get_context() to retrieve stored conversation information
2. Check the returned values:
   - current_driver: The driver being discussed
   - current_compound: The tyre compound being used
   - current_tyre_life: Recent tyre life value

DECISION LOGIC:
- If current_driver is NOT None/null → USE IT, don't ask for it
- If current_compound is NOT None/null → USE IT, don't ask for it
- If current_tyre_life is NOT None/null → USE IT as reference
- ONLY ask the user if ALL context values are None/null

WORKFLOW:
Step 1: ALWAYS call get_context() first
Step 2: Extract available context values
Step 3: If driver + compound available → call tyre_data_tool(driver=X, compound=Y, last_n=5)
Step 4: Analyze tyre degradation and provide strategy recommendations
Step 5: If needed, update context with store_context

Analysis guidelines:
1. Compare lap times across increasing tyre life to identify degradation
2. Identify which sectors are degrading first
3. Assess performance window status
4. Recommend: Push / Manage / Pit soon / Pit now

EXAMPLES:

Example 1 - Follow-up question:
User: "what about his tyre life?"
→ get_context() returns: {current_driver: 'RUS', current_compound: 'HARD', ...}
→ Use RUS and HARD directly
→ tyre_data_tool(driver='RUS', compound='HARD', last_n=5)
→ Analyze and respond WITHOUT asking for clarification

Example 2 - No context:
User: "analyze tyre degradation"
→ get_context() returns: {current_driver: None, current_compound: None, ...}
→ Ask: "Which driver and compound should I analyze?"

CRITICAL RULE: If context exists, USE IT. Never ask for information you already have.
""",
        tools=[tyre_data_tool, store_context, get_context]
    )

if 'race_context_agent' not in st.session_state:
    st.session_state.race_context_agent = LlmAgent(
        name='RaceContextAgent',
        model=GEMINI_MODEL,
        instruction="""You are a Formula 1 Race Context Analysis Agent.

CONTEXT AWARENESS:
1. FIRST call get_context() to check for stored driver information
2. If current_driver exists in context, use it
3. Only ask for clarification if no context is available

Your role:
- Explain what is happening around the car during the race
- Focus on track conditions, position changes, and contextual anomalies
- Do NOT analyze tyre degradation or strategy decisions

Workflow:
1. Call get_context() first
2. Extract driver from context if available
3. Call race_context_tool with the driver
4. Analyze position changes, track status, and anomalies
5. Provide concise race-engineer style output
""",
        tools=[race_context_tool, store_context, get_context]
    )

# Chief Engineer with enhanced context handling
if 'chief_engineer' not in st.session_state:
    st.session_state.chief_engineer = LlmAgent(
        name='ChiefEngineer',
        instruction="""You are the Chief Race Engineer with conversation memory.

CRITICAL PRE-ROUTING STEPS:
1. IMMEDIATELY call get_context() to retrieve stored conversation state
2. Analyze the user's question for context clues:
   - Pronouns: "he/his/him" = current_driver from context
   - References: "his tyres", "the pace", "that driver" = use context
   - Implicit subjects: "what about tyre life?" = current_driver from context

3. ROUTING LOGIC with context injection:

   FOR PACE/PERFORMANCE QUESTIONS:
   - Route to PerformanceAnalysisAgent
   - Agent will automatically store context after analysis

   FOR TYRE/STRATEGY QUESTIONS:
   - Check if context has current_driver + current_compound
   - If YES: Route to TyreStrategyAgent (it will use get_context internally)
   - If NO: Route to TyreStrategyAgent (it will ask for clarification)

   FOR POSITION/TRACK STATUS QUESTIONS:
   - Route to RaceContextAgent
   - Pass context if available

4. PRONOUN RESOLUTION EXAMPLES:

   Conversation 1:
   User: "How is RUS's pace?"
   → Route to PerformanceAnalysisAgent
   → Context stored: driver='RUS', compound='HARD'
   
   User: "what about his tyre life?"
   → get_context() shows: current_driver='RUS', current_compound='HARD'
   → Route to TyreStrategyAgent
   → Agent will automatically use RUS + HARD from context
   
   Conversation 2:
   User: "Analyze VER"
   → get_context() shows: current_driver='VER'
   
   User: "should he pit?"
   → "he" = VER from context
   → Route to TyreStrategyAgent with confidence it has context

ROUTING RULES:
- LAP TIMES, SECTOR SPEEDS, PACE → PerformanceAnalysisAgent
- TYRES, DEGRADATION, PIT STOPS, STRATEGY → TyreStrategyAgent
- POSITIONS, TRACK STATUS, INCIDENTS → RaceContextAgent

CRITICAL: The specialist agents have get_context() and will use it. Your job is to:
1. Check context exists via get_context()
2. Route to the right specialist
3. Trust the specialist to use context properly
4. Summarize their findings

Never manually resolve context - let the specialist agents call get_context().
""",
        model=GEMINI_MODEL,
        tools=[
            get_context,
            AgentTool(agent=st.session_state.performance_analysis_agent),
            AgentTool(agent=st.session_state.tyre_strategy_agent),
            AgentTool(agent=st.session_state.race_context_agent)
        ]
    )

# Initialize session service
if 'session_service' not in st.session_state:
    st.session_state.session_service = InMemorySessionService()

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = f"{SESSION_ID_PREFIX}{uuid.uuid4()}"

if 'runner' not in st.session_state:
    st.session_state.runner = Runner(
        agent=st.session_state.chief_engineer,
        app_name=APP_NAME,
        session_service=st.session_state.session_service
    )

if 'session_initialized' not in st.session_state:
    import asyncio
    
    async def init_session():
        await st.session_state.session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=st.session_state.session_id
        )
    
    try:
        asyncio.run(init_session())
        st.session_state.session_initialized = True
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(init_session())
        loop.close()
        st.session_state.session_initialized = True

st.set_page_config(
    page_title="F1 AI Mission Control",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the "Engineering Console" look
st.title("F1 MISSION CONTROL",text_alignment="center")
st.markdown("""
<style>
    /* Main Background - Dark Slate */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Neon Accents */
    h1, h2, h3 {
        font-family: 'Roboto Mono', monospace;
        color: #00FF9D !important; /* Petronas Green / Data Green */
    }
    
    /* Chat Bubbles - Tech style */
    .stChatMessage {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 4px;
        font-family: 'Source Code Pro', monospace;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363D;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1F242D;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #FF385C; /* F1 Red */
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Show current context
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=100) # Or your hackathon team logo
    st.title("🎛️ STRATEGY HUD")
    
    st.markdown("---")
    
    # Create placeholders for live updating
    driver_metric = st.empty()
    compound_metric = st.empty()
    life_metric = st.empty()
    status_metric = st.empty()

    st.markdown("---")
    st.caption("SYSTEM DIAGNOSTICS")
    st.code(f"Session: {st.session_state.session_id[-8:]}\nLatency: 12ms\nConnection: SECURE")

    # Function to render metrics with color coding
    def render_sidebar():
        ctx = st.session_state.get('conversation_context', {})
        
        # Driver Metric
        drv = ctx.get('current_driver', 'N/A')
        driver_metric.metric("Target Driver", drv, border=True)
        
        # Compound (Color Coded Logic)
        comp = ctx.get('current_compound', 'N/A')
        comp_color = "normal"
        if "SOFT" in str(comp).upper(): comp_color = "off" # Red indicator
        
        compound_metric.metric("Tyre Compound", comp)
        
        # Tyre Life with Delta indicator logic (mock logic for demo)
        life = ctx.get('current_tyre_life', 0)
        life_val = f"{life} Laps" if life else "N/A"
        # If life > 20, show red delta (warning)
        delta_color = "inverse" if life and life > 20 else "normal"
        life_metric.metric("Tyre Age", life_val, delta_color=delta_color)
        
        # Analysis Status
        last_act = ctx.get('last_analysis_type', 'Ready')
        status_metric.info(f"AGENT STATUS: {str(last_act).upper()}")

    render_sidebar()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ex: 'How is VER's pace?' then 'what about his tyre life?'"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Chief Engineer coordinating..."):
            try:
                runner = st.session_state.runner
                
                user_message = types.Content(
                    role="user",
                    parts=[types.Part(text=prompt)]
                )

                response_generator = runner.run(
                    new_message=user_message,
                    session_id=st.session_state.session_id,
                    user_id=USER_ID
                )
                
                for event in response_generator:
                    if hasattr(event, 'text') and event.text:
                        full_response += str(event.text)
                        response_placeholder.markdown(full_response + "▌")
                    elif hasattr(event, 'content') and hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                full_response += str(part.text)
                                response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"System Error: {str(e)}")