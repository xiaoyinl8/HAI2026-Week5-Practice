import streamlit as st
import json
from pydantic import BaseModel, Field
from typing import Optional, List
from movie_tool import get_tools, query_movie_db
from chart_tool import get_chart_tool, validate_chart


# ── State ──

DEFAULT_STATE = {
    "agent_phase": "idle",
    "agent_events": [],
    "agent_messages": [],
    "agent_tools": [],
    "agent_df": None,
    "agent_chart_specs": [],
    "agent_pending_message": None,
    "agent_alternatives": [],
    "selected_alternative": None,
}

def get_state(key):
    return st.session_state.get(key, DEFAULT_STATE[key])

def set_state(key, value):
    st.session_state[key] = value

def restart_agent(user_question, filtered_df, show_chart=False):
    set_state("agent_phase", "thinking")
    set_state("agent_events", [])
    set_state("agent_chart_specs", [])
    set_state("agent_pending_message", None)
    set_state("agent_alternatives", [])
    set_state("selected_alternative", None)

    tools = get_tools(filtered_df)
    system_content = "You are a data analyst with access to a tool that executes Python code on a movie database."

    if show_chart:
        tools.append(get_chart_tool())
        system_content += " After computing the data, create a chart using a Vega-Lite specification."

    set_state("agent_messages", [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_question},
    ])
    set_state("agent_tools", tools)
    set_state("agent_df", filtered_df)


# ── Logic ──

def run_step(client):
    phase = get_state("agent_phase")
    messages = get_state("agent_messages")

    if phase == "thinking":
        class Reasoning(BaseModel):
            reason: str = Field(description="Your reasoning about what you know so far and what to do next")
            use_tool: bool = Field(description="True if you need to run code or create a chart, False if you can give the final answer")
            answer: Optional[str] = Field(default=None, description="Your final answer in one short paragraph. Only provide when use_tool is False.")

        response = client.chat.completions.parse(
            model="gpt-4o-mini", messages=messages, response_format=Reasoning,
        )
        reasoning = response.choices[0].message.parsed
        messages.append({"role": "assistant", "content": reasoning.reason})

        if reasoning.use_tool:
            get_state("agent_events").append({"type": "thought", "thought": reasoning.reason})
            set_state("agent_phase", "acting")
        else:
            get_state("agent_events").append({"type": "answer", "thought": reasoning.reason, "answer": reasoning.answer})
            set_state("agent_phase", "done")

    elif phase == "acting":
        tools = get_state("agent_tools")

        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=tools, parallel_tool_calls=False,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            set_state("agent_phase", "done")
            return

        set_state("agent_pending_message", msg)
        set_state("agent_phase", "awaiting_approval")

def execute_pending_tools():
    messages = get_state("agent_messages")
    df = get_state("agent_df")
    pending_msg = get_state("agent_pending_message")

    messages.append(pending_msg)
    for tc in pending_msg.tool_calls:
        args = json.loads(tc.function.arguments)

        if tc.function.name == "QueryMovieDB":
            result = query_movie_db(args["code"], df)
            get_state("agent_events").append({
                "type": "action", "name": tc.function.name,
                "code": args["code"], "result": result,
            })
        elif tc.function.name == "CreateChart":
            spec, result = validate_chart(args["vega_lite_spec"])
            if spec:
                get_state("agent_chart_specs").append(spec)
            get_state("agent_events").append({
                "type": "chart", "name": tc.function.name,
                "spec_str": args["vega_lite_spec"], "result": result,
            })

        messages.append({"role": "tool", "content": result, "tool_call_id": tc.id})

    set_state("agent_pending_message", None)
    set_state("agent_phase", "thinking")

def reject_pending_tools(feedback):
    messages = get_state("agent_messages")
    pending_msg = get_state("agent_pending_message")

    rejection_msg = "User rejected this action."
    if feedback:
        rejection_msg += f" User feedback: {feedback}"

    messages.append(pending_msg)
    for tc in pending_msg.tool_calls:
        get_state("agent_events").append({
            "type": "rejected", "name": tc.function.name,
            "feedback": feedback,
        })
        messages.append({"role": "tool", "content": rejection_msg, "tool_call_id": tc.id})

    set_state("agent_pending_message", None)
    set_state("agent_phase", "generating_alternatives")


def generate_alternatives(client):
    """Generate multiple alternative approaches when user rejects an action"""
    messages = get_state("agent_messages")
    
    class Alternative(BaseModel):
        title: str = Field(description="A short, descriptive title for this approach (5-10 words)")
        description: str = Field(description="A detailed explanation of this alternative approach and how it differs from the rejected action")
        rationale: str = Field(description="Why this approach might work better or address the user's concerns")
    
    class Alternatives(BaseModel):
        alternatives: List[Alternative] = Field(
            description="3 meaningfully different alternative approaches the agent could take to solve the user's question",
            min_length=3,
            max_length=3
        )
    
    # Add a system message requesting alternatives
    alt_messages = messages + [{
        "role": "user",
        "content": "The user rejected your proposed action. Generate 3 different alternative approaches to answer their question. Each approach should use a meaningfully different strategy, tool usage pattern, or analysis method. Focus on diversity of approaches rather than minor variations."
    }]
    
    response = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=alt_messages,
        response_format=Alternatives,
    )
    
    alternatives = response.choices[0].message.parsed.alternatives
    set_state("agent_alternatives", alternatives)
    get_state("agent_events").append({
        "type": "alternatives_generated",
        "alternatives": alternatives,
    })
    set_state("agent_phase", "selecting_alternative")


def apply_selected_alternative(alternative_index):
    """Apply the user's selected alternative approach"""
    messages = get_state("agent_messages")
    alternatives = get_state("agent_alternatives")
    
    if 0 <= alternative_index < len(alternatives):
        selected = alternatives[alternative_index]
        set_state("selected_alternative", alternative_index)
        
        # Add user's selection as guidance
        guidance_msg = f"User selected approach: '{selected.title}'. {selected.description}. Please proceed with this strategy."
        messages.append({"role": "user", "content": guidance_msg})
        
        get_state("agent_events").append({
            "type": "alternative_selected",
            "alternative": selected,
            "index": alternative_index,
        })
        
        set_state("agent_phase", "thinking")


# ── Rendering ──

def render_events():
    for event in get_state("agent_events"):
        if event["type"] == "thought":
            st.markdown(f"**Thought:** {event['thought']}")
        elif event["type"] == "action":
            st.markdown(f"**Action:** `{event['name']}`")
            st.code(event["code"], language="python")
            st.markdown("**Observation:**")
            st.code(event["result"], language="text")
            st.divider()
        elif event["type"] == "chart":
            st.markdown(f"**Action:** `{event['name']}`")
            st.code(event["spec_str"], language="json")
            st.markdown("**Observation:**")
            st.code(event["result"], language="text")
            st.divider()
        elif event["type"] == "rejected":
            st.markdown(f"**Rejected:** `{event['name']}`")
            if event.get("feedback"):
                st.text(f"Feedback: {event['feedback']}")
            st.divider()
        elif event["type"] == "alternatives_generated":
            st.markdown("**Agent generated alternative approaches**")
            st.divider()
        elif event["type"] == "alternative_selected":
            alt = event["alternative"]
            st.markdown(f"**User selected:** {alt.title}")
            st.divider()
        elif event["type"] == "answer":
            st.markdown(f"**Thought:** {event['thought']}")

def render_pending_approval():
    st.warning("The agent wants to perform the following action:")
    for tc in get_state("agent_pending_message").tool_calls:
        args = json.loads(tc.function.arguments)
        st.markdown(f"**Tool:** `{tc.function.name}`")
        if tc.function.name == "QueryMovieDB":
            st.code(args["code"], language="python")
        elif tc.function.name == "CreateChart":
            st.code(args["vega_lite_spec"], language="json")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        approved = st.button("Approve", type="primary", use_container_width=True)
    with btn_col2:
        rejected = st.button("Reject", use_container_width=True)
    return approved, rejected

def render_pending_feedback():
    st.info("Provide optional feedback to help guide alternative approaches:")
    feedback = st.text_area(
        "What's wrong with this approach? What would you prefer instead?",
        key="reject_feedback",
        placeholder="e.g., 'Too complex, use a simpler calculation' or 'Focus on genre instead of year'",
        height=100
    )
    submitted = st.button("Generate Alternative Approaches", type="primary", use_container_width=True)
    return submitted, feedback


def render_alternative_selection():
    """Render UI for selecting from alternative approaches"""
    st.success("The agent has generated 3 alternative approaches. Select one to proceed:")
    
    alternatives = get_state("agent_alternatives")
    
    # Display each alternative as an expandable card
    for i, alt in enumerate(alternatives):
        with st.expander(f"**Option {i+1}: {alt.title}**", expanded=(i == 0)):
            st.markdown(f"**Approach:** {alt.description}")
            st.markdown(f"**Rationale:** {alt.rationale}")
            if st.button(f"Select Option {i+1}", key=f"select_alt_{i}", use_container_width=True, type="primary" if i == 0 else "secondary"):
                return i
    
    return None


def render_panel():
    st.subheader("Analysis Results")
    container = st.container(height=600)
    actions = {}
    with container:
        phase = get_state("agent_phase")

        if phase == "idle":
            st.info("Enter a question and click 'Analyze' to see results.")

        elif phase in ("thinking", "acting"):
            with st.expander("Agent Reasoning Trace", expanded=True):
                render_events()
            st.spinner("Agent is thinking...")

        elif phase == "awaiting_approval":
            with st.expander("Agent Reasoning Trace", expanded=True):
                render_events()
            approved, rejected = render_pending_approval()
            actions = {"approved": approved, "rejected": rejected}

        elif phase == "awaiting_feedback":
            with st.expander("Agent Reasoning Trace", expanded=True):
                render_events()
            submitted, feedback = render_pending_feedback()
            actions = {"submitted": submitted, "feedback": feedback}
        
        elif phase == "generating_alternatives":
            with st.expander("Agent Reasoning Trace", expanded=True):
                render_events()
            st.spinner("Generating alternative approaches...")
        
        elif phase == "selecting_alternative":
            with st.expander("Agent Reasoning Trace", expanded=True):
                render_events()
            selected = render_alternative_selection()
            actions = {"selected_alternative": selected}

        elif phase == "done":
            with st.expander("Agent Reasoning Trace", expanded=False):
                render_events()
            events = get_state("agent_events")
            if events and events[-1].get("answer"):
                st.write("**Answer:**")
                st.write(events[-1]["answer"])
            for spec in get_state("agent_chart_specs"):
                st.vega_lite_chart(spec, use_container_width=True)

    return actions


# ── Lifecycle ──

def agent_panel(client, analyze_button, user_question, filtered_df, show_chart=False):
    # Phases: idle -> thinking <-> acting -> awaiting_approval -> awaiting_feedback 
    #         -> generating_alternatives -> selecting_alternative -> thinking ... -> done
    if analyze_button and user_question:
        restart_agent(user_question, filtered_df, show_chart)

    actions = render_panel()

    phase = get_state("agent_phase")
    if phase in ("thinking", "acting"):
        run_step(client)
        st.rerun()
    elif phase == "awaiting_approval":
        if actions.get("approved"):
            execute_pending_tools()
            st.rerun()
        elif actions.get("rejected"):
            set_state("agent_phase", "awaiting_feedback")
            st.rerun()
    elif phase == "awaiting_feedback" and actions.get("submitted"):
        reject_pending_tools(actions.get("feedback", ""))
        st.rerun()
    elif phase == "generating_alternatives":
        generate_alternatives(client)
        st.rerun()
    elif phase == "selecting_alternative" and actions.get("selected_alternative") is not None:
        apply_selected_alternative(actions.get("selected_alternative"))
        st.rerun()
