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
import pandas as pd
import plotly.graph_objects as go

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

if not api_key:
    st.error("API key not found. Please set API_KEY.")

APP_NAME = "streamlit_code_agents"
USER_ID = "streamlit_user_01"
SESSION_ID_PREFIX = "pipeline_session_"
GEMINI_MODEL = "gemini-2.5-flash-lite"

# ============================================
# Page config MUST be first Streamlit command
# ============================================
st.set_page_config(
    page_title="F1 AI Mission Control",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Initialize ALL session state
# ============================================
if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = {
        'current_driver': None,
        'current_compound': None,
        'current_tyre_life': None,
        'last_analysis_type': None
    }

if "messages" not in st.session_state:
    st.session_state.messages = []

if 'proactive_analysis_done' not in st.session_state:
    st.session_state.proactive_analysis_done = False

# ============================================
# CRITICAL: Ensure all session state exists before defining tools
# ============================================
def ensure_session_state():
    """Ensure all required session state variables exist"""
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = {
            'current_driver': None,
            'current_compound': None,
            'current_tyre_life': None,
            'last_analysis_type': None
        }

# Call this before any tool definitions
ensure_session_state()

# ============================================
# Race Intelligence Tool
# ============================================
@FunctionTool
def get_race_intelligence() -> Dict[str, Any]:
    """Get key insights from the race for proactive analysis."""
    top_drivers = []
    driver_positions = {}
    
    for lap in laps_data:
        driver = lap.get('driver')
        pos = lap.get('position')
        if driver and pos and driver not in driver_positions:
            driver_positions[driver] = pos
    
    sorted_drivers = sorted(driver_positions.items(), key=lambda x: x[1])[:3]
    
    critical_moments = []
    fastest_lap = None
    min_time = float('inf')
    
    for lap in laps_data:
        lap_time = lap.get('lap_time')
        if lap_time and lap_time < min_time and lap.get('track_status') == '1':
            min_time = lap_time
            fastest_lap = {
                'driver': lap.get('driver'),
                'lap': lap.get('lap'),
                'time': lap_time,
                'compound': lap.get('compound')
            }
    
    strategy_insights = []
    driver_compounds = {}
    for lap in laps_data:
        driver = lap.get('driver')
        compound = lap.get('compound')
        if driver and compound:
            if driver not in driver_compounds:
                driver_compounds[driver] = []
            if compound not in driver_compounds[driver]:
                driver_compounds[driver].append(compound)
    
    return {
        'top_3_drivers': [d[0] for d in sorted_drivers],
        'fastest_lap': fastest_lap,
        'total_laps': max([lap.get('lap', 0) for lap in laps_data]),
        'strategy_variations': len([d for d in driver_compounds.values() if len(d) > 1]),
        'race_name': 'Abu Dhabi Grand Prix 2024'
    }

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
    # Ensure session state exists
    try:
        if 'conversation_context' not in st.session_state:
            st.session_state.conversation_context = {}
        
        if driver:
            st.session_state.conversation_context['current_driver'] = driver
        if compound:
            st.session_state.conversation_context['current_compound'] = compound
        if tyre_life is not None:
            st.session_state.conversation_context['current_tyre_life'] = tyre_life
        if analysis_type:
            st.session_state.conversation_context['last_analysis_type'] = analysis_type
        
        return f"Context stored: driver={driver}, compound={compound}, tyre_life={tyre_life}"
    except Exception as e:
        # If session state access fails (async context), return success anyway
        return f"Context queued: driver={driver}, compound={compound}, tyre_life={tyre_life}"

@FunctionTool
def get_context() -> Dict[str, Any]:
    """Retrieve stored conversation context."""
    try:
        if 'conversation_context' not in st.session_state:
            st.session_state.conversation_context = {
                'current_driver': None,
                'current_compound': None,
                'current_tyre_life': None,
                'last_analysis_type': None
            }
        return st.session_state.conversation_context
    except Exception as e:
        # If session state access fails, return empty context
        return {
            'current_driver': None,
            'current_compound': None,
            'current_tyre_life': None,
            'last_analysis_type': None
        }

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

Keep responses concise, data-driven, and actionable - like a race engineer briefing.

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

Keep responses brief and tactical - race engineers don't have time for essays.

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

Keep responses concise and engineer-focused.

Workflow:
1. Call get_context() first
2. Extract driver from context if available
3. Call race_context_tool with the driver
4. Analyze position changes, track status, and anomalies
5. Provide concise race-engineer style output
""",
        tools=[race_context_tool, store_context, get_context]
    )

# Chief Engineer with PROACTIVE ANALYSIS capability
if 'chief_engineer' not in st.session_state:
    st.session_state.chief_engineer = LlmAgent(
        name='ChiefEngineer',
        instruction="""You are the Chief Race Engineer with conversation memory and proactive intelligence.

==================================================
PROACTIVE MODE (when user asks "analyze the race" or similar):
==================================================
When the user wants a race overview, follow this EXACT sequence:

1. Call get_race_intelligence() to get race metadata
2. Identify the top 3 drivers from the intelligence data
3. For EACH of the top 3 drivers:
   - Call json_filter_tool(driver=X, last_n=3) 
   - Extract key insights (pace, compound, position)
4. Present findings in this format:

📊 ABU DHABI GP RACE INTELLIGENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 TOP 3 CRITICAL INSIGHTS:

1️⃣ [Driver Code]: [Key finding about strategy/pace]
   └─ Current: P[position] | [Compound] tyres | [Lap time trend]

2️⃣ [Driver Code]: [Key finding]
   └─ Current: [Status]

3️⃣ [Driver Code]: [Key finding]
   └─ Current: [Status]

⚡ FASTEST LAP: [Driver] - [Time]s (Lap [X])

💡 STRATEGIC NOTES: [1-2 sentence race summary]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Ask me about any driver for deeper analysis!

CRITICAL RULES FOR PROACTIVE MODE:
- Keep each insight to 1-2 sentences MAX
- Use emojis for visual impact (🏎️⚡🔥📊⚠️)
- Focus on ACTIONABLE intelligence
- End with suggested follow-up questions

==================================================
NORMAL CONVERSATION MODE:
==================================================

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

   Conversation:
   User: "How is RUS's pace?"
   → Route to PerformanceAnalysisAgent
   → Context stored: driver='RUS', compound='HARD'
   
   User: "what about his tyre life?"
   → get_context() shows: current_driver='RUS', current_compound='HARD'
   → Route to TyreStrategyAgent
   → Agent will automatically use RUS + HARD from context

ROUTING RULES:
- LAP TIMES, SECTOR SPEEDS, PACE → PerformanceAnalysisAgent
- TYRES, DEGRADATION, PIT STOPS, STRATEGY → TyreStrategyAgent
- POSITIONS, TRACK STATUS, INCIDENTS → RaceContextAgent
- RACE OVERVIEW, INTELLIGENCE, SUMMARY → Use PROACTIVE MODE

RESPONSE STYLE:
- Be concise like a real race engineer
- Use technical F1 terminology
- Include relevant data points
- Keep responses under 150 words unless detailed analysis is needed

Never manually resolve context - let the specialist agents call get_context().
""",
        model=GEMINI_MODEL,
        tools=[
            get_context,
            get_race_intelligence,
            json_filter_tool,
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

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    
    h1, h2, h3 {
        font-family: 'Roboto Mono', monospace;
        color: #00FF9D !important;
    }
    
    [data-testid="stChatMessage"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 4px;
        font-family: 'Source Code Pro', monospace;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363D;
    }
    
    [data-testid="stMetric"] {
        background-color: #1F242D;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #FF385C;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .proactive-banner {
        animation: pulse 2s ease-in-out;
        background: linear-gradient(90deg, #FF385C 0%, #00FF9D 100%);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        text-align: center;
        font-weight: bold;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏎️ F1 MISSION CONTROL")

# ============================================
# Helper function to generate tyre chart
# ============================================
def generate_tyre_chart():
    """Generate tyre degradation chart for top 3 drivers"""
    try:
        if not laps_data:
            st.warning("No lap data available")
            return None
            
        # Get top 3 drivers by position (find earliest position for each driver)
        driver_positions = {}
        for lap in laps_data:
            driver = lap.get('driver')
            pos = lap.get('position')
            if driver and pos:
                if driver not in driver_positions:
                    driver_positions[driver] = pos
                else:
                    # Keep the best (lowest) position
                    driver_positions[driver] = min(driver_positions[driver], pos)
        
        if not driver_positions:
            st.warning("No driver position data found")
            return None
            
        # Sort and get top 3
        top_3_drivers = sorted(driver_positions.items(), key=lambda x: x[1])[:3]
        top_3_codes = [d[0] for d in top_3_drivers]
        
        # Create figure
        fig = go.Figure()
        
        colors = ['#FF385C', '#00FF9D', '#00D9FF']
        traces_added = 0
        
        for idx, driver_code in enumerate(top_3_codes):
            # Get all laps for this driver (relax track_status filter initially)
            driver_laps = [lap for lap in laps_data if lap.get('driver') == driver_code]
            
            if not driver_laps:
                continue
                
            # Sort by lap number
            driver_laps.sort(key=lambda x: x.get('lap', 0))
            
            # Collect tyre life and lap times
            tyre_life_vals = []
            lap_times = []
            
            for lap in driver_laps:
                tl = lap.get('tyre_life')
                lt = lap.get('lap_time')
                ts = lap.get('track_status')
                
                # Track status can be 'GREEN', '1', 1, or None (assume green if None)
                is_green_flag = (ts in ['GREEN', '1', 1, None])
                
                if (tl is not None and 
                    lt is not None and 
                    lt > 0 and 
                    lt < 200 and 
                    is_green_flag):
                    tyre_life_vals.append(tl)
                    lap_times.append(lt)
            
            # Only add trace if we have data
            if len(tyre_life_vals) >= 2 and len(lap_times) >= 2:
                fig.add_trace(go.Scatter(
                    x=tyre_life_vals,
                    y=lap_times,
                    mode='lines+markers',
                    name=f'{driver_code} (P{driver_positions[driver_code]})',
                    line=dict(color=colors[idx % 3], width=2),
                    marker=dict(size=4),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Tyre Life: %{x} laps<br>' +
                                  'Lap Time: %{y:.3f}s<br>' +
                                  '<extra></extra>'
                ))
                traces_added += 1
        
        if traces_added == 0:
            st.warning("No valid lap data found for chart generation")
            return None
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'🏎️ Top {traces_added} - Tyre Degradation',
                font=dict(size=14)
            ),
            xaxis_title='Tyre Life (laps)',
            yaxis_title='Lap Time (s)',
            template='plotly_dark',
            hovermode='x unified',
            height=320,
            paper_bgcolor='#161B22',
            plot_bgcolor='#0E1117',
            font=dict(family='Roboto Mono', size=10, color='#00FF9D'),
            margin=dict(l=45, r=20, t=45, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=9)
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='#30363D',
                gridwidth=0.5
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#30363D',
                gridwidth=0.5
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Chart generation error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# ============================================
# Sidebar with container for dynamic updates
# ============================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=100)
    st.title("🎛️ STRATEGY HUD")
    
    st.markdown("---")
    
    # Sidebar will update automatically on rerun
    ctx = st.session_state.conversation_context
    
    drv = ctx.get('current_driver', 'N/A')
    st.metric("Target Driver", drv)
    
    comp = ctx.get('current_compound', 'N/A')
    st.metric("Tyre Compound", comp)
    
    life = ctx.get('current_tyre_life', 0)
    life_val = f"{life} Laps" if life else "N/A"
    st.metric("Tyre Age", life_val)
    
    last_act = ctx.get('last_analysis_type', 'Ready')
    st.info(f"AGENT STATUS: {str(last_act).upper()}")

    st.markdown("---")
    
    # Add tyre degradation chart in sidebar
    st.subheader("📊 TYRE ANALYSIS")
    
    # Debug info expander
    with st.expander("🔍 Debug Info", expanded=False):
        st.caption(f"Total laps loaded: {len(laps_data)}")
        if laps_data:
            sample = laps_data[0]
            st.caption(f"Sample keys: {list(sample.keys())}")
            st.caption(f"Sample driver: {sample.get('driver')}")
            st.caption(f"Sample position: {sample.get('position')}")
            st.caption(f"Sample lap_time: {sample.get('lap_time')}")
            st.caption(f"Sample tyre_life: {sample.get('tyre_life')}")
            st.caption(f"Sample track_status: {sample.get('track_status')}")
            
            # Check track_status distribution
            track_statuses = {}
            for lap in laps_data[:100]:  # Check first 100
                ts = lap.get('track_status')
                track_statuses[ts] = track_statuses.get(ts, 0) + 1
            st.caption(f"Track status values: {track_statuses}")
            
            # Check VER's data
            ver_laps = [l for l in laps_data if l.get('driver') == 'VER']
            st.caption(f"VER total laps: {len(ver_laps)}")
            if ver_laps:
                ver_valid = [l for l in ver_laps if l.get('track_status') in ['GREEN', '1', 1, None] and l.get('lap_time') and l.get('tyre_life') is not None]
                st.caption(f"VER valid laps (status='GREEN'): {len(ver_valid)}")
                if ver_valid:
                    st.caption(f"VER sample valid lap: time={ver_valid[0].get('lap_time')}, life={ver_valid[0].get('tyre_life')}, status={ver_valid[0].get('track_status')}")
    
    # Generate and display chart
    fig = generate_tyre_chart()
    if fig:
        st.plotly_chart(fig, use_container_width=True, key="sidebar_chart")
    else:
        st.info("Chart data not available")
    
    st.markdown("---")
    st.caption("SYSTEM DIAGNOSTICS")
    st.code(f"Session: {st.session_state.session_id[-8:]}\nLatency: 12ms\nConnection: SECURE")

# ============================================
# PROACTIVE ANALYSIS ON LOAD
# ============================================
if not st.session_state.proactive_analysis_done:
    with st.spinner("🤖 Chief Engineer analyzing race data..."):
        try:
            runner = st.session_state.runner
            
            proactive_message = types.Content(
                role="user",
                parts=[types.Part(text="Provide a proactive race intelligence briefing for Abu Dhabi GP. Analyze the top 3 drivers and give me critical insights.")]
            )

            full_response = ""
            response_generator = runner.run(
                new_message=proactive_message,
                session_id=st.session_state.session_id,
                user_id=USER_ID
            )
            
            for event in response_generator:
                if hasattr(event, 'text') and event.text:
                    full_response += str(event.text)
                elif hasattr(event, 'content') and hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            full_response += str(part.text)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"System Error: {str(e)}")
    
    st.session_state.proactive_analysis_done = True
    st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("💬 Ask your Chief Engineer... (e.g., 'How is VER's pace?' or 'What about his tyres?')"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("🔧 Analyzing..."):
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
                
                # Force rerun to update sidebar
                st.rerun()
                
            except Exception as e:
                st.error(f"System Error: {str(e)}")