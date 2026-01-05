from google.genai import types
import json
from dotenv import load_dotenv
import os
import streamlit as st
from google.adk.agents.llm_agent import LlmAgent
from typing import Dict, List, Optional, Any
from google.adk.tools import FunctionTool,ToolContext
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
GEMINI_MODEL = "gemini-2.5-flash"


@FunctionTool
def json_filter_tool(driver:str,
                    team:Optional[str]=None,
                    track_status:str=None,
                    last_n=5,
                    lap:Optional[int]=None,
                    compound:Optional[str]=None)-> List[dict]:
    
    # Filter by driver
    filtered = [lap_data for lap_data in laps_data if lap_data["driver"] == driver]

    # Filter by team
    if team:
        filtered = [lap_data for lap_data in filtered if lap_data["team"].upper() == team.upper()]

    # Filter by track status
    if track_status:
        filtered = [lap_data for lap_data in filtered if lap_data.get("track_status", "").upper() == track_status.upper()]

    # Filter by specific lap
    if lap is not None:
        filtered = [lap_data for lap_data in filtered if lap_data.get("lap") == lap]

    # Filter by tyre compound
    if compound:
        filtered = [lap_data for lap_data in filtered if lap_data.get("compound", "").upper() == compound.upper()]

    # Select only the last N laps if multiple remain
    if last_n and len(filtered) > last_n:
        filtered = filtered[-last_n:]

    return filtered


if 'performance_analysis_agent' not in st.session_state:
    st.session_state.performance_analysis_agent = LlmAgent(
        name='PerformanceAnalysisAgent',
        model=GEMINI_MODEL,
        instruction="""You are a Formula 1 Performance Analysis Agent.

        Task:
        - Analyze the performance of a driver in a race.
        - Use the tool `json_filter_tool` to access race data (laps_clean.json). 
        - Provide insights only based on the filtered data returned by the tool.

        Instructions for using the tool:
        - Pass these parameters to `json_filter_tool`:
        - driver: the driver code to analyze (e.g., VER, HAM)
        - team: optional, if analyzing a team
        - lap: optional, specific lap to analyze
        - last_n: optional, number of recent laps to analyze (default 5)
        - track_status: optional, e.g., GREEN
        - compound: optional, tyre compound

        - Once you get the filtered data, analyze it and give insights about:
        1. Overall pace trend (improving, stable, degrading)
        2. Sector strengths and weaknesses
        3. Tyre condition and strategy recommendations
        4. Track context or anomalies if applicable

        - Only use the data returned from the tool. Do not make up lap times or positions.

        Example response format:
        - "Driver VER is showing improving pace over the last 5 laps. Sector 2 is slightly slower than average, while Sector 1 and 3 are strong. Tyres are MEDIUM with 2 laps of life left. Track conditions are GREEN. No pit stops expected immediately.""",
        tools=[json_filter_tool]
    )

@FunctionTool
def tyre_data_tool(
    driver: str,
    compound: Optional[str] = None,
    tyre_life: Optional[int] = None,
    last_n: int = 5,
    track_status: Optional[str] = None
) -> List[dict]:
    """
    Tool for extracting tyre-related telemetry data for strategy analysis.
    """

    # 1. Filter by driver
    filtered = [lap for lap in laps_data if lap["driver"] == driver]

    # 2. Filter by compound
    if compound:
        filtered = [
            lap for lap in filtered
            if lap.get("compound", "").upper() == compound.upper()
        ]

    # 3. Filter by exact tyre life (if user asks for a specific lap age)
    if tyre_life is not None:
        filtered = [
            lap for lap in filtered
            if lap.get("tyre_life") == tyre_life
        ]

    # 4. Filter by track status
    if track_status:
        filtered = [
            lap for lap in filtered
            if lap.get("track_status", "").upper() == track_status.upper()
        ]

    # 5. Sort by lap number (critical for degradation analysis)
    filtered.sort(key=lambda x: x.get("lap", 0))

    # 6. Return last N laps
    if last_n and len(filtered) > last_n:
        filtered = filtered[-last_n:]

    return filtered



if not 'tyre_strategy_agent' in st.session_state:
    st.session_state.tyre_strategy_agent = LlmAgent(
        name='TyreStrategyAgent',
        model=GEMINI_MODEL,
        instruction="""You are a Formula 1 Tyre & Strategy Agent.

        Your role:
        - Analyze tyre performance, degradation, and pit strategy.
        - Base your reasoning ONLY on lap telemetry returned by the tool.
        - You do NOT invent data or assume missing laps.

        You have access to the tool `tyre_data_tool`.

        How to use the tool:
        - Extract relevant parameters from the user query:
        - driver (required)
        - compound (optional)
        - tyre_life (optional, exact value if mentioned)
        - last_n (default 5 if user asks about recent behaviour)
        - track_status (optional)

        - Call `tyre_data_tool` to retrieve lap telemetry.
        - Use ONLY the returned data for analysis.

        Available telemetry fields:
        - tyre_life
        - fresh_tyre
        - lap_time
        - sector_times
        - position
        - compound
        - track_status

        Analysis guidelines:
        1. Determine tyre degradation by comparing lap times across increasing tyre life.
        2. Identify which sector(s) are degrading first.
        3. Assess whether the tyre is still in a performance window or dropping off.
        4. Recommend:
        - Push
        - Tyre management
        - Pit window (early / optimal / extend)

        Rules:
        - Do NOT fabricate lap times or sector data.
        - If data is insufficient, clearly state that.
        - Keep conclusions concise and race-engineer style.

        Example output:
        "VER's MEDIUM tyres show mild degradation over the last 5 laps. Sector 2 times are increasing while Sector 1 remains stable. Tyre life is at 6 laps, still within the performance window. Recommend extending the stint by 2–3 laps unless traffic increases.""",
                tools=[tyre_data_tool]
    )

@FunctionTool
def race_context_tool(
    driver: str,
    team: Optional[str] = None,
    track_status: Optional[str] = None,
    last_n: int = 5
) -> List[dict]:
    """
    Tool to extract race context information such as
    track status, position changes, and lap anomalies.
    """

    # 1. Filter by driver
    filtered = [lap for lap in laps_data if lap["driver"] == driver]

    # 2. Optional team filter
    if team:
        filtered = [
            lap for lap in filtered
            if lap.get("team", "").upper() == team.upper()
        ]

    # 3. Optional track status filter
    if track_status:
        filtered = [
            lap for lap in filtered
            if lap.get("track_status", "").upper() == track_status.upper()
        ]

    # 4. Sort chronologically
    filtered.sort(key=lambda x: x.get("lap", 0))

    # 5. Take last N laps
    if last_n and len(filtered) > last_n:
        filtered = filtered[-last_n:]

    # 6. Return ONLY race-context-relevant fields
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

if not 'race_context_agent' in st.session_state:
    st.session_state.race_context_agent = LlmAgent(
        name = 'RaceContextAgent',
        model=GEMINI_MODEL,
        instruction="""You are a Formula 1 Race Context Analysis Agent.

        Your role:
        - Explain what is happening around the car during the race.
        - Focus on track conditions, position changes, and contextual anomalies.
        - Do NOT analyze tyre degradation or strategy decisions.

        You have access to the tool `race_context_tool`.

        How to use the tool:
        - Extract relevant parameters from the user question:
        - driver (required)
        - team (optional)
        - track_status (optional)
        - last_n (number of recent laps to analyze; default 5)

        - Call `race_context_tool` with these parameters.
        - Use ONLY the data returned by the tool for reasoning.

        Available data fields:
        - lap
        - position
        - lap_time
        - sector_times
        - track_status
        - driver
        - team

        Analysis guidelines:
        1. Identify changes in track status (GREEN, YELLOW, etc.).
        2. Detect lap-time anomalies and correlate them with track conditions.
        3. Analyze position changes across laps.
        4. Determine whether performance changes are caused by:
        - Track conditions
        - Traffic
        - Race incidents
        - Normal race evolution

        Rules:
        - Do NOT invent safety cars, incidents, or penalties.
        - Do NOT assume tyre-related effects.
        - If the data is insufficient, clearly state that.

        Output style:
        - Concise
        - Race-engineer tone
        - Cause → effect explanation

        Example output:
        "VER lost one position on lap 14. Lap time increased by ~3.2s, aligned with a YELLOW track status in Sector 2. Pace drop is contextual rather than performance-related. No abnormal race conditions detected in the following lap."
        """,
        tools=[race_context_tool]
    )

#--- Supervisor Agent Logic ---


if 'chief_engineer' not in st.session_state:
    st.session_state.chief_engineer = LlmAgent(
        name='ChiefEngineer',
        instruction="""You are the Chief Race Engineer. Your job is to route user queries to the correct specialist.

- If the user asks about LAP TIMES, SECTOR SPEEDS, or OVERALL PACE: Route to 'PerformanceAnalysisAgent'.
- If the user asks about TYRES, DEGRADATION, PIT STOPS, or STRATEGY: Route to 'TyreStrategyAgent'.
- If the user asks about POSITIONS, TRACK STATUS (Yellow/Green), or INCIDENTS: Route to 'RaceContextAgent'.

Always summarize the final answer provided by the specialists into a concise report for the driver.
""",
        model=GEMINI_MODEL,
        # We give the supervisor access to the other agents as tools
        tools=[
            AgentTool(agent=st.session_state.performance_analysis_agent),
            AgentTool(agent=st.session_state.tyre_strategy_agent),
            AgentTool(agent=st.session_state.race_context_agent)
        ]
    )

# Initialize session service ONCE and reuse it
if 'session_service' not in st.session_state:
    st.session_state.session_service = InMemorySessionService()

# Generate a unique session ID for this browser tab/user session
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = f"{SESSION_ID_PREFIX}{uuid.uuid4()}"

# Runner Initialization (Only Once)
if 'runner' not in st.session_state:
    st.session_state.runner = Runner(
        agent=st.session_state.chief_engineer,
        app_name=APP_NAME,
        session_service=st.session_state.session_service
    )

# Initialize session using asyncio (properly handle async function)
if 'session_initialized' not in st.session_state:
    import asyncio
    
    async def init_session():
        await st.session_state.session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=st.session_state.session_id
        )
    
    # Run the async function
    try:
        asyncio.run(init_session())
        st.session_state.session_initialized = True
    except RuntimeError:
        # If event loop is already running, use different approach
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(init_session())
        loop.close()
        st.session_state.session_initialized = True

    

# --- Streamlit UI ---

st.title("🏎️ F1 AI Mission Control")
st.caption(f"Connected to Session: {st.session_state.session_id}")
st.markdown("_Chief Engineer Online. Systems: Performance, Strategy, and Context linked._")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle Input
if prompt := st.chat_input("Ex: 'How is VER's pace compared to his tyre life?'"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Chief Engineer coordinating with specialists..."):
            try:
                runner = st.session_state.runner
                
                # The Runner returns a generator of events
                user_message = types.Content(
                    role="user",
                    parts=[types.Part(text=prompt)]
                )

                response_generator = runner.run(
                    new_message=user_message,
                    session_id=st.session_state.session_id,
                    user_id=USER_ID
                )
                full_response = ""
                # Iterate over the generator to get the response text
                for event in response_generator:
                    if hasattr(event, 'text') and event.text:
                        full_response += str(event.text)
                        response_placeholder.markdown(full_response + "▌")
                    elif hasattr(event, 'content') and hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                full_response += str(part.text)
                                response_placeholder.markdown(full_response + "▌")
                    
                    

                # Final update to remove the cursor
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"System Error: {str(e)}")