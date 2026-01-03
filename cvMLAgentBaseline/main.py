#!/usr/bin/env python3
"""
CV ML Agent for Gymnastics Analytics
An agentic ML workflow that loads external models and tools for comprehensive analysis.

Features:
- Face detection
- Weight/body shape detection
- Pose estimation (multiple models)
- Person detection
- Technique supervised metrics
- FIG standard reasoning from memory indexes
- ML model selection with reasoning on sequence of steps
"""

import logging
import asyncio
import sys
import os
import json
import re
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, date

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('cvMLAgent_debug.log')
    ]
)

logger = logging.getLogger(__name__)

# Import vision-agents framework
try:
    from vision_agents.core import User, Agent, cli
    from vision_agents.core.agents import AgentLauncher
    from vision_agents.plugins import getstream, openai, ultralytics
    VISION_AGENTS_AVAILABLE = True
except ImportError:
    logger.warning("vision-agents not available - running in standalone mode")
    VISION_AGENTS_AVAILABLE = False

# Import our custom modules
from model_selector import MLModelSelector, ModelReasoning
from cv_tools import (
    FaceDetectionTool,
    WeightDetectionTool,
    PoseEstimationTool,
    PersonDetectionTool
)
from technique_metrics import TechniqueSupervisedMetrics
from memory_integration import MemoryIndexIntegration
from ml_workflow import AgenticMLWorkflow
from deep_search_reasoning import DeepSearchReasoning
from dynamic_metric_discovery import DynamicMetricDiscovery
from user_request_handler import UserRequestHandler

load_dotenv()

# Global instances
model_selector: Optional[MLModelSelector] = None
cv_tools: Dict[str, Any] = {}
workflow: Optional[AgenticMLWorkflow] = None
memory_integration: Optional[MemoryIndexIntegration] = None
deep_search: Optional[DeepSearchReasoning] = None
metric_discovery: Optional[DynamicMetricDiscovery] = None
request_handler: Optional[UserRequestHandler] = None


async def create_agent(**kwargs):
    """
    Create the CV ML agent with:
    - ML model selection with reasoning
    - Multiple CV tools (face, weight, pose, person detection)
    - Technique supervised metrics
    - Memory integration for FIG standards
    - Agentic ML workflow
    """
    global model_selector, cv_tools, workflow, memory_integration
    
    logger.info("🤖 Initializing CV ML Agent...")
    
    # Initialize LLM instance first (required for memory integration and deep search)
    llm_instance = None
    
    if VISION_AGENTS_AVAILABLE:
        # Try Gemini Realtime first (preferred for video)
        GEMINI_AVAILABLE = False
        try:
            from vision_agents.plugins import gemini
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")
            if GEMINI_API_KEY:
                GEMINI_AVAILABLE = True
                try:
                    llm_instance = gemini.Realtime(fps=3)
                    logger.info("✅ Using Gemini Realtime for video processing")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to initialize Gemini Realtime: {e}")
                    logger.warning("   Falling back to OpenAI...")
                    llm_instance = None
            else:
                logger.warning("⚠️  Gemini plugin available but no API key found (GEMINI_API_KEY or GOOGLE_AI_API_KEY)")
        except ImportError:
            logger.warning("⚠️  Gemini plugin not available - will try OpenAI Realtime")
        
        # Fallback to OpenAI Realtime if Gemini not available or failed
        if llm_instance is None:
            logger.info("🔍 Attempting OpenAI fallback...")
            try:
                # Check if openai.Realtime exists
                if hasattr(openai, 'Realtime'):
                    logger.info("   Found openai.Realtime, attempting to initialize...")
                    try:
                        llm_instance = openai.Realtime(fps=3)
                        logger.info("✅ Using OpenAI Realtime for video processing")
                    except Exception as e:
                        logger.error(f"❌ Failed to initialize openai.Realtime: {e}")
                        logger.error(f"   Error type: {type(e).__name__}")
                        raise  # Re-raise to trigger next fallback
                else:
                    logger.warning("⚠️  openai.Realtime not found in vision_agents.plugins.openai")
                    logger.warning("   Available openai attributes: " + ", ".join([attr for attr in dir(openai) if not attr.startswith('_')][:10]))
                    
                    # Try using OpenAI LLM with vision model as last resort
                    # NOTE: This will likely fail with "video capability required" error
                    logger.warning("⚠️  Attempting openai.LLM as last resort (this will likely fail)...")
                    try:
                        llm_instance = openai.LLM(model="gpt-4o-mini")
                        logger.warning("⚠️  Using OpenAI LLM (not Realtime) - this may cause validation errors")
                    except Exception as e:
                        logger.error(f"❌ openai.LLM failed: {e}")
                        raise  # Re-raise to show final error
                        
            except Exception as e:
                logger.error("=" * 60)
                logger.error("❌ ALL LLM OPTIONS FAILED")
                logger.error("=" * 60)
                logger.error("The vision-agents framework requires a Realtime LLM for video processing.")
                logger.error("")
                logger.error("Available options:")
                logger.error("  1. ✅ Gemini Realtime (recommended):")
                logger.error("     - Set GEMINI_API_KEY in .env file")
                logger.error("     - Get key from: https://makersuite.google.com/app/apikey")
                logger.error("")
                logger.error("  2. ❓ OpenAI Realtime:")
                logger.error("     - Check if your vision-agents version supports openai.Realtime")
                logger.error("     - Ensure OPENAI_API_KEY is set in .env file")
                logger.error("")
                logger.error("  3. ❌ OpenAI LLM (does NOT work):")
                logger.error("     - openai.LLM() does NOT provide video capability")
                logger.error("     - Will fail with: 'video capability required' error")
                logger.error("=" * 60)
                raise ValueError(
                    "Video capability requires a Realtime LLM. "
                    "Please set GEMINI_API_KEY or OPENAI_API_KEY in .env file. "
                    "See ENV_SETUP.md for instructions."
                ) from e
        
        # Final check - ensure we have an LLM instance
        if llm_instance is None:
            raise ValueError(
                "Failed to initialize LLM. "
                "Please set GEMINI_API_KEY or OPENAI_API_KEY in .env file."
            )
    
    # Initialize model selector with reasoning
    model_selector = MLModelSelector()
    logger.info("✅ Created ML Model Selector")
    
    # Initialize memory integration (now llm_instance is available)
    memory_integration = MemoryIndexIntegration(use_deep_research=True, llm_instance=llm_instance)
    logger.info("✅ Created Memory Index Integration")
    
    # Initialize deep search for metric discovery (now llm_instance is available)
    global deep_search, metric_discovery
    deep_search = DeepSearchReasoning(llm_instance=llm_instance)
    metric_discovery = DynamicMetricDiscovery(deep_search, memory_integration)
    logger.info("✅ Initialized Deep Search & Metric Discovery")
    
    # Initialize CV tools
    cv_tools['face_detection'] = FaceDetectionTool()
    cv_tools['weight_detection'] = WeightDetectionTool()
    cv_tools['pose_estimation'] = PoseEstimationTool()
    cv_tools['person_detection'] = PersonDetectionTool()
    logger.info(f"✅ Initialized {len(cv_tools)} CV tools")
    
    # Initialize technique supervised metrics
    technique_metrics = TechniqueSupervisedMetrics(memory_integration)
    logger.info("✅ Initialized Technique Supervised Metrics")
    
    # Initialize agentic ML workflow
    workflow = AgenticMLWorkflow(
        model_selector=model_selector,
        cv_tools=cv_tools,
        technique_metrics=technique_metrics,
        memory_integration=memory_integration
    )
    logger.info("✅ Initialized Agentic ML Workflow")
    
    # Initialize user request handler
    global request_handler
    request_handler = UserRequestHandler(metric_discovery, workflow, memory_integration)
    logger.info("✅ Initialized User Request Handler")
    
    # Get selected technique from kwargs
    selected_technique = kwargs.get("technique", None)
    user_requests = kwargs.get("user_requests", [])
    
    # Get activity from kwargs - DEFAULT TO GYMNASTICS for cvMLAgentNew
    selected_activity = kwargs.get("activity", "gymnastics")  # Automatically focus on gymnastics
    
    # Build agent instructions - GYMNASTICS-FOCUSED on ACL metrics
    instructions = f"""You are a CV ML Agent specialized in **GYMNASTICS ACL RISK ANALYSIS**.

**PRIMARY FOCUS: ACL TEAR RISK ANALYSIS**
- Analyze gymnastics techniques (back handsprings, flips, vaults, bars, beam, floor routines)
- **Primary focus**: ACL tear risk metrics during landing phases
- Frame capture is **always active** - analysis runs continuously
- ACL risk is calculated in real-time during performance

Your capabilities:
1. **ML Model Selection**: Automatically select and reason about which models to use based on:
   - Activity type (gymnastics, fitness, posture, yoga)
   - User requests (tracking height off floor, impact force, landing bend angles, stiffness, knee straightness, etc.)
   - Technique requirements
   - Available computational resources
   - Model performance characteristics

2. **CV Tools Available**:
   - Face Detection: Detect and track athlete faces
   - Weight Detection: Estimate body weight/shape from pose
   - Pose Estimation: Multiple models (HRNet, YOLO, MediaPipe)
   - Person Detection: Detect and track athletes in video

3. **ACL Risk Metrics** (Primary Focus):
   - ACL tear risk score (0-1, higher = higher risk)
   - ACL risk level (MINIMAL/LOW/MODERATE/HIGH)
   - Knee valgus angle (knock-knee collapse)
   - Knee flexion during landing (shock absorption)
   - Impact force (landing force)
   - Asymmetric landing risk
   - HIGH risk timesteps flagged automatically

4. **Standard Reasoning**: Use existing memory indexes to:
   - Retrieve standards for techniques (FIG for gymnastics, biomechanical for posture, etc.)
   - Compare metrics to standards
   - Provide reasoning on compliance

5. **Agentic ML Workflow**: 
   - Reason about sequence of steps
   - Select appropriate models for each step
   - Chain model outputs intelligently
   - Provide comprehensive analysis

**Current Activity**: Gymnastics (ACL risk analysis)
**Current Technique**: {selected_technique or "Detected automatically"}
**Focus**: ACL tear risk metrics during landing phases

**IMPORTANT - ACL RISK ANALYSIS:**
1. **Frame capture is always active** - analysis runs continuously
2. **ACL risk is calculated automatically** during landing phases
3. **Focus on ACL metrics** - provide analysis on landing technique and ACL risk factors
4. **Be concise** - don't repeat what the system is doing, just provide ACL risk analysis results

**Workflow**:
1. When user provides activity, technique, and requests, reason about:
   - Which models are needed (pose, face, person detection, etc.)
   - Sequence of processing steps
   - How to combine model outputs
   - Activity-specific base metrics to track

2. **ACL Risk Metric Calculation**:
   - ACL risk metrics calculated automatically during landing phases
   - Focus analysis on ACL risk factors: valgus collapse, landing stiffness, impact force
   - Report HIGH risk moments when detected (score >= 0.7)

3. Execute ACL risk analysis:
   - Process frames continuously (always active)
   - Calculate ACL risk metrics during landing phases
   - Flag HIGH risk moments automatically
   - Provide concise ACL risk analysis

**Frame Capture:**
- Always active - no start/stop needed
- ACL risk analysis runs continuously during performance

4. **CRITICAL - Analysis Format with Root Causes and Recommendations:**

When providing analysis, you MUST structure your response with clear sections:

**A. ACL RISK ANALYSIS:**
- **CRITICAL**: ONLY report metrics that EXIST in `workflow.workflow_state["metrics"]`
- **NEVER** make up, estimate, or infer metric values
- **NEVER** report metrics that haven't been calculated yet
- Check if metric exists before reporting: `if "metric_name" in workflow.workflow_state.get("metrics", {{}}):`
- If metric doesn't exist, state: "Metric not yet calculated" or "Waiting for frame processing"
- Report HIGH risk moments (score >= 0.7) with specific frame/timestamp
- Use ONLY actual numbers from workflow.workflow_state["metrics"]

**B. ACL RISK FACTORS:**
For HIGH risk moments, identify contributing factors:
- **Valgus collapse**: Knee collapsing inward (knock-knee pattern)
- **Stiff landing**: Insufficient knee flexion (< 15°)
- **High impact**: Excessive landing force
- **Asymmetric landing**: Uneven weight distribution

Format: "**Root Cause for [metric]**: [clear explanation of WHY the deviation occurs]"

**C. ACL RISK REDUCTION RECOMMENDATIONS:**
For HIGH risk moments, provide specific landing technique corrections:

Format:
"**Priority [1/2/3]: [What to fix]**
- **Why**: [ACL risk factor]
- **How**: [Specific landing technique correction]
- **Impact**: [How this reduces ACL risk]"

Example:
"**Priority 1: Increase Landing Knee Flexion**
- **Why**: Landing knee bend is 12° (too stiff) - ideal is 15-30° for shock absorption
- **How**: Focus on soft landing - bend knees more on impact, land with weight centered
- **Impact**: Reduces ACL tear risk by improving shock absorption"

**D. PRIORITY RANKING:**
Rank by ACL risk severity:
1. **Priority 1**: HIGH ACL risk (score >= 0.7) - immediate landing technique corrections needed
2. **Priority 2**: MODERATE ACL risk (score 0.4-0.7) - landing improvements recommended
3. **Priority 3**: Preventive landing technique refinements

**E. CONCISE ACL RISK RESPONSE FORMAT:**

**FIRST: Check if metrics exist before formatting response:**
```
metrics = workflow.workflow_state.get("metrics", {{}})
if not metrics or len(metrics) == 0:
    return "No metrics calculated yet. Waiting for frame processing to begin analysis."
```

**Then format response ONLY if metrics exist:**

```
⚠️ **ACL RISK ANALYSIS**

**Risk Level**: [ONLY if "acl_risk_level" exists in metrics]
**Overall Risk Score**: [ONLY if "acl_tear_risk_score" exists in metrics]

**HIGH RISK MOMENTS** (ONLY if in acl_flagged_timesteps array):
- Frame [X] at [X.XX]s - Score: X.XX
  - Risk Factors: [ONLY use values from flagged_timestep data]
  - Correction: [Specific landing technique fix]

**PRIORITY RECOMMENDATIONS**:
1. [Most critical ACL risk reduction]
2. [Secondary landing improvement]

**SUMMARY**: [Brief ACL risk assessment]
```

**CRITICAL**: If any metric doesn't exist, don't include it in the response. Only report what's actually calculated.

**F. CRITICAL GUIDELINES - NO HALLUCINATION:**
- **NEVER report metrics that don't exist** - always check `workflow.workflow_state.get("metrics", {{}})` first
- **NEVER make up, estimate, or infer metric values** - only use actual calculated values
- **If metrics dictionary is empty**: State "No metrics calculated yet. Waiting for frame processing."
- **If specific metric missing**: State "Metric '[name]' not yet calculated" - do NOT estimate it
- **Be concise** - don't repeat what the system is doing
- Focus on ACL risk metrics and landing technique
- Report HIGH risk moments immediately when detected (ONLY if in `acl_flagged_timesteps`)
- Provide specific landing technique corrections
- Use ONLY actual metric values from workflow state (verify they exist first)
- Don't explain system processes - just provide ACL risk analysis

**Examples of user requests to detect:**
- "I want to track height and landing angles" (gymnastics)
- "I want to improve my knee straightness" (gymnastics)
- "Track impact force and improve landing technique" (gymnastics)
- "Check my squat depth and knee alignment" (fitness)
- "Analyze my sitting posture" (posture)
- "Improve my Warrior I hip alignment" (yoga)

When you detect these, respond with: "🔍 METRIC_DISCOVERY:[full_user_request]"

**IMPORTANT - Follow-Up Queries and Revisions:**
You MUST actively listen to and respond to ALL follow-up queries, suggestions, and revisions from the user:
- "Actually, I want to track X instead" → Update metrics immediately
- "Can you also measure Y?" → Add Y to tracked metrics
- "Forget about Z, focus on W" → Remove Z, add W
- "Change technique to [new technique]" → Research new technique and update workflow
- "What about [different aspect]?" → Research and add relevant metrics

**When user provides revisions:**
1. Acknowledge the change: "Got it, I'll update to track X instead of Y"
2. Trigger research if needed: "🔍 METRIC_DISCOVERY:[revised_request]" or "🔍 TECHNIQUE_RESEARCH:[new_technique]"
3. Update workflow with new metrics/technique
4. Confirm the update: "✅ Updated! Now tracking: [list of metrics]"

**IMPORTANT - Unknown Technique Handling:**
If a technique is NOT in your mapping or you don't know about it:
1. **Immediately trigger deep research**: Respond with "🔍 TECHNIQUE_RESEARCH:[technique_name]"
2. **Research the technique**: Use deep research to discover:
   - Technique standards and biomechanics
   - Key metrics for that technique
   - Common errors and corrections
   - Activity-specific requirements
3. **Derive appropriate metrics**: Based on research findings, determine what metrics to track
4. **Update workflow**: Initialize workflow with discovered technique and metrics
5. **Inform user**: "I've researched [technique] and will track: [metrics list]"

**Technique Research Process:**
- When you see "🔍 TECHNIQUE_RESEARCH:[technique]", the system will:
  1. Search web for technique standards and biomechanics
  2. Use LLM reasoning to extract key metrics
  3. Determine appropriate measurement methods
  4. Update workflow with discovered technique and metrics

**IMPORTANT - Explicit Metric Queries:**
When users explicitly ask about specific metrics, you MUST provide the actual values from frame inference:
- "What is the height off ground?" → Provide the calculated height value (e.g., "Your height off floor is 0.85 meters" or "850 pixels")
- "How many rotations did I do?" → Provide the rotation count (e.g., "You completed 1.5 rotations" or "1 full rotation with 0.3 partial rotation")
- "What are the angles?" → Provide the joint angles (e.g., "Knee angles: left 165°, right 162°; Hip angles: left 145°, right 148°")
- "What's my knee angle?" → Provide the specific knee angle value (e.g., "Your left knee angle is 165°")

**CRITICAL - Accessing Metric Values (NO HALLUCINATION):**
The workflow stores calculated metrics in `workflow.workflow_state["metrics"]` after processing frames. 

**STRICT RULES - VERIFY BEFORE REPORTING:**
1. **ALWAYS check if metrics dictionary exists and is not empty**:
   ```
   metrics = workflow.workflow_state.get("metrics", {{}})
   if not metrics or len(metrics) == 0:
       # NO METRICS CALCULATED YET - say so, don't make anything up
       return "No metrics calculated yet. Waiting for frame processing."
   ```

2. **ALWAYS check if specific metric exists** before reporting:
   ```
   if "metric_name" not in metrics:
       # METRIC DOESN'T EXIST - say so, don't estimate
       return "Metric 'metric_name' not yet calculated."
   ```

3. **NEVER make up metric values** - only use values that exist in the dictionary
4. **NEVER estimate or infer** metric values based on other information
5. **NEVER assume** a metric exists just because it should be calculated
6. **NEVER use placeholder values** like "approximately X" or "around Y" - only report actual calculated values

**When answering explicit queries:**
1. **First**: Check if `workflow.workflow_state.get("metrics", {{}})` exists and is not empty
   - If empty: "No metrics calculated yet. Waiting for frame processing."
2. **Second**: Check if the specific metric key exists in the metrics dictionary
   - If missing: "Metric '[name]' not yet calculated."
3. **Third**: Extract the value ONLY if it exists
   - Use actual value: `metrics["metric_name"]`
4. **Fourth**: Format the response with the actual value and units
5. **NEVER**: Make up, estimate, or infer values - only report what actually exists

**Metric Names to Look For:**
- Height: `height_off_floor_pixels`, `height_off_floor_meters`, `height_off_floor_normalized`
- Rotations: `rotation_count`, `current_rotation_angle`, `partial_rotation`
- Angles: `knee_angle_left`, `knee_angle_right`, `hip_angle_left`, `hip_angle_right`, `elbow_angle_left`, `elbow_angle_right`, `shoulder_angle_left`, `shoulder_angle_right`, `ankle_angle_left`, `ankle_angle_right`
- **ACL Risk**: `acl_tear_risk_score` (0-1, higher = higher risk), `acl_risk_level` (MINIMAL/LOW/MODERATE/HIGH), `acl_max_valgus_angle`, `acl_insufficient_flexion_risk`, `acl_high_impact_risk`, `acl_asymmetric_landing_risk`
- All joint angles are available when "angle" or "angles" is requested

The system calculates these metrics from recorded frame inference data. Always reference the actual metric values when answering explicit queries.

**ALWAYS PROVIDE ROOT CAUSES AND RECOMMENDATIONS:**
- Never just report metrics - always explain WHY deviations occur (root cause)
- Always provide specific, actionable recommendations for improvement
- Use the structured format above for clear, professional analysis
- Access metrics from `workflow.workflow_state["metrics"]` to provide concrete values
- Access standards comparison from `workflow.workflow_state.get("analysis", {{}}).get("comparison", {{}})` for gaps and recommendations

**ACL RISK REPORTING:**
1. **ACL Risk is always calculated** for gymnastics
2. **Report HIGH risk moments** (score >= 0.7) immediately with specific frame/timestamp
3. **Don't report MODERATE/LOW risk** unless specifically asked
4. **Focus on actionable landing corrections** for HIGH risk moments
   
3. **If ACL risk is calculated (gymnastics or requested) AND HIGH risk is detected:**
   - **CRITICAL**: ONLY report ACL risk timesteps that are ACTUALLY in `workflow.workflow_state.get("acl_flagged_timesteps", [])`
   - **DO NOT** estimate, calculate, or infer HIGH risk moments - ONLY use what's in the flagged_timesteps array
   - **DO NOT** report frame numbers or risk scores that are not explicitly in the flagged_timesteps array
   - Access `workflow.workflow_state.get("acl_flagged_timesteps", [])` to get all HIGH risk timesteps
   - Access `workflow.workflow_state.get("landing_phases", [])` to get detected landing phases
   - For EACH flagged timestep, provide SPECIFIC details:
     - **Frame/Timestamp**: Exact frame number and timestamp from the flagged_timestep entry (e.g., "Frame 70 at 2.34s")
     - **Risk Score**: The ACL risk score from the flagged_timestep entry (0.7-1.0 for HIGH risk)
     - **Landing Context**: If `landing_context` is available, mention:
       - "Detected during landing phase (confidence: X%)"
       - "Landing phase type: [pre_landing/impact/post_impact]"
       - "Landing indicators: [list indicators]"
     - **Why HIGH Risk**: Use `primary_risk_factors` from flagged timestep data to explain:
       - "Knee valgus collapse (12.5°) - indicates knock-knee landing pattern"
       - "Insufficient knee flexion (15.0°) - landing too stiff, insufficient shock absorption"
       - "High impact force (3200N) - excessive landing force"
     - **Biomechanical Explanation**: Explain WHY each factor increases ACL tear risk
     - **Specific Frame Context**: Reference the specific frame number and what was happening at that moment
     - **Video Clip**: Mention that a video clip has been saved for this frame (if available)
   
4. **Format for HIGH risk timesteps:**
   ```
   ⚠️ **ACL HIGH RISK DETECTED**
   
   **Frame [X] at [timestamp]s** (Risk Score: X.XX)
   - Landing Phase: [If detected during landing, mention it]
   - Primary Risk Factors:
     • [Factor 1 with value] - [Why this increases ACL risk]
     • [Factor 2 with value] - [Why this increases ACL risk]
   - Biomechanical Analysis: [Detailed explanation of why this combination is dangerous]
   - Video Clip: [Clip filename if available]
   - Recommendation: [Immediate action to reduce risk]
   ```
   
5. **If ACL risk is calculated (gymnastics or requested) but NO HIGH risk detected:**
   - **First check**: Verify `acl_tear_risk_score` exists in metrics
   - **If metric exists**: State "ACL risk analysis completed. No HIGH risk detected. Risk level: [use actual value from metrics]"
   - **If metric doesn't exist**: State "ACL risk metrics not yet calculated. Waiting for frame processing."
   - Do NOT list all timesteps if they're not HIGH risk
   - **DO NOT** report specific frame numbers or risk scores if `acl_flagged_timesteps` is empty
   - **DO NOT** estimate or infer HIGH risk moments based on other metrics (valgus angles, knee flexion, etc.)
   - **DO NOT** make up risk levels - only use values that exist in metrics
   - Only report what is explicitly stored in `acl_flagged_timesteps` array

**WHAT TO SAY WHEN DATA IS MISSING:**
- **If metrics dictionary is empty**: "No metrics calculated yet. Waiting for frame processing."
- **If specific metric missing**: "Metric '[name]' not yet calculated."
- **If acl_flagged_timesteps is empty**: "No HIGH risk moments detected."
- **If asked about metric that doesn't exist**: "Metric '[name]' not yet calculated. It will be available after frame processing."
- **NEVER**: Make up values, estimate, or say "approximately X" - only report actual calculated values

**When Analysis is Available:**
- **First check**: Verify `workflow.workflow_state.get("metrics", {{}})` exists and is not empty
- **If metrics empty**: State "No metrics calculated yet. Analysis will begin after frame processing."
- **If metrics exist**: Check `workflow.workflow_state.get("acl_flagged_timesteps", [])` for HIGH risk moments
- **Only report HIGH risk moments** that are actually in the `acl_flagged_timesteps` array
- **Only use metric values** that actually exist in the metrics dictionary
- Report HIGH risk moments with specific frame/timestamp and risk factors (from actual data)
- Provide concise landing technique corrections
- Don't repeat system processes - just provide ACL risk analysis

**CRITICAL - ACL Risk Data Validation:**
- **NEVER** report specific frame numbers, timestamps, or risk scores for ACL risk unless they are EXACTLY in `workflow.workflow_state.get("acl_flagged_timesteps", [])`
- **NEVER** estimate, calculate, or infer HIGH risk moments based on other metrics (valgus angles, knee flexion, impact force, etc.)
- **NEVER** report "Frame X at Y seconds - HIGH Risk (Score: Z)" unless that exact entry exists in `acl_flagged_timesteps`
- If `acl_flagged_timesteps` is empty or has no entries, you MUST state: "No HIGH risk ACL moments detected (all risk scores were below 0.7 threshold)"
- If you see aggregated metrics showing valgus angles, knee flexion, or impact forces that seem concerning, you can mention them in recommendations, but DO NOT report them as HIGH risk timesteps unless they're in `acl_flagged_timesteps`

**BE CONCISE - NO HALLUCINATION:**
- **NEVER report metrics that don't exist** in `workflow.workflow_state["metrics"]`
- **Always verify metric exists** before reporting: check dictionary keys first
- **If no metrics yet**: State "Waiting for frame processing to calculate metrics"
- Don't explain what the system is doing (frame capture, processing, etc.)
- Don't repeat metric calculations or system processes
- Don't make up metric values - only use actual calculated values
- Focus only on ACL risk analysis results (from actual data)
- Provide brief, actionable landing technique feedback
"""
    
    # Note: We use PoseEstimationTool (HRNet) instead of vision-agents processors
    # The workflow will use HRNet for pose estimation via cv_tools
    # Processors list is empty - we handle pose estimation manually with HRNet
    processors = []
    if VISION_AGENTS_AVAILABLE:
        # Create MetricsProcessor to process frames and calculate metrics
        try:
            from cvMLAgent.metrics_processor import MetricsProcessor
            
            print("🟡 [main.py] Creating MetricsProcessor...")
            # Create metrics processor that will process frames and calculate metrics
            metrics_processor = MetricsProcessor(workflow, cv_tools)
            processors.append(metrics_processor)
            print(f"🟡 [main.py] ✅ MetricsProcessor created and added to processors list (total: {len(processors)})")
            logger.info("✅ Created MetricsProcessor for frame processing and metric calculation")
        except Exception as e:
            print(f"🟡 [main.py] ❌ ERROR creating MetricsProcessor: {e}")
            logger.warning(f"⚠️  Could not create MetricsProcessor: {e}")
            logger.warning("   Metrics calculation may not work during video stream")
        
        # Optional: Also add YOLO processor for pose overlay visualization
        logger.info("ℹ️  Using PoseEstimationTool (HRNet) for pose detection")
        try:
            from vision_agents.plugins import ultralytics
            if hasattr(ultralytics, 'YOLOPoseProcessor'):
                yolo_processor = ultralytics.YOLOPoseProcessor(
                    model_path="yolo11n-pose.pt",
                    conf_threshold=0.5
                )
                processors.append(yolo_processor)
                logger.info("✅ Added YOLO processor for pose overlay visualization")
        except Exception as e:
            logger.debug(f"⚠️  Could not create YOLO processor (optional): {e}")
    
    # Note: llm_instance was already initialized earlier in this function
    
    if not VISION_AGENTS_AVAILABLE:
        logger.info("ℹ️  Running in standalone mode (no vision-agents)")
        # Return a mock agent object for standalone mode
        class MockAgent:
            def __init__(self):
                self.processors = []
        return MockAgent()
    
    # Check for required Stream API credentials
    stream_api_key = os.getenv("STREAM_API_KEY")
    stream_api_secret = os.getenv("STREAM_API_SECRET")
    
    if not stream_api_key or not stream_api_secret:
        logger.error("=" * 60)
        logger.error("❌ MISSING STREAM API CREDENTIALS")
        logger.error("=" * 60)
        logger.error("The agent requires Stream API credentials to run.")
        logger.error("")
        logger.error("Please create a .env file in cvMLAgent/ with:")
        logger.error("  STREAM_API_KEY=your_api_key")
        logger.error("  STREAM_API_SECRET=your_api_secret")
        logger.error("")
        logger.error("You can copy from videoAgent/.env or get new keys from:")
        logger.error("  https://getstream.io/")
        logger.error("")
        logger.error("Example .env file location: cvMLAgent/.env")
        logger.error("=" * 60)
        raise ValueError(
            "Missing Stream API credentials. "
            "Please set STREAM_API_KEY and STREAM_API_SECRET in .env file. "
            "See env_template.txt for an example."
        )
    
    print(f"🟡 [main.py] Creating Agent with {len(processors)} processor(s)...")
    for i, proc in enumerate(processors):
        proc_name = getattr(proc, 'name', type(proc).__name__)
        print(f"🟡 [main.py]    Processor {i+1}: {proc_name}")
    
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="CV ML Gymnastics Analyst", id="cv_ml_agent"),
        instructions=instructions,
        llm=llm_instance,
        processors=processors
    )
    
    print(f"🟡 [main.py] ✅ Agent created successfully with {len(processors)} processor(s)")
    logger.info("✅ CV ML Agent created successfully")
    return agent


async def join_call(agent, call_type: str, call_id: str, **kwargs) -> None:
    """
    Join the call and start the CV ML agent workflow.
    """
    global workflow, model_selector, memory_integration, request_handler
    
    if not agent:
        logger.error("❌ Agent not initialized")
        return
    
    call = await agent.create_call(call_type, call_id)
    
    logger.info("🤸 Starting CV ML Agent...")
    
    # Get activity, technique, and user requests
    selected_activity = kwargs.get("activity", "gymnastics")  # Default to gymnastics
    selected_technique = kwargs.get("technique", None)
    user_requests = kwargs.get("user_requests", [])
    
    # If user requests provided, discover metrics via deep research
    if user_requests and metric_discovery:
        try:
            discovery = metric_discovery.discover_metrics_from_request(
                user_text=" ".join(user_requests),
                technique=selected_technique,
                activity=selected_activity
            )
            # Use discovered metrics instead of raw requests
            user_requests = discovery.get("metrics_to_track", user_requests)
            logger.info(f"✅ Discovered {len(user_requests)} metrics from deep research")
        except Exception as e:
            logger.warning(f"⚠️  Metric discovery failed, using provided requests: {e}")
    
    # Initialize workflow with activity, technique, and requests
    # For gymnastics, enable frame capture automatically
    if workflow:
        # Frame capture always active for ACL analysis
        workflow.workflow_state["frame_capture_active"] = True
        logger.info("✅ Frame capture always active for ACL risk analysis")
        
        if selected_technique:
            await workflow.initialize_workflow(
                technique=selected_technique,
                user_requests=user_requests
            )
            logger.info(f"✅ Workflow initialized for activity: {selected_activity}, technique: {selected_technique}")
        else:
            # Initialize with ACL-focused metrics
            # Technique will be detected automatically from video
            default_metrics = [
                "ACL tear risk",
                "landing bend angles",
                "impact force"
            ]
            await workflow.initialize_workflow(
                technique="unknown",  # Will be detected automatically
                user_requests=user_requests or default_metrics
            )
            logger.info(f"✅ Workflow initialized for gymnastics (technique will be detected automatically)")
            print("🤸 [Gymnastics] Workflow initialized - technique will be detected automatically from video")
    
    # Store call_id in workflow for transcript saving
    if workflow:
        workflow.set_call_id(call_id)
        logger.info(f"📝 Call ID set: {call_id}")
    
    # Variables to track activity/technique (may be updated during call)
    selected_activity = kwargs.get("activity", "gymnastics")  # Default to gymnastics
    selected_technique = kwargs.get("technique", None)
    
    # Set up event handlers to capture ALL user messages and agent responses
    # This ensures complete transcript capture
    try:
        # Try to subscribe to user speech transcription events if available
        # Check for vision-agents event system
        if hasattr(agent, 'on_realtime_user_speech_transcription'):
            @agent.on_realtime_user_speech_transcription
            async def on_user_transcript(event):
                """Capture all user speech transcriptions"""
                user_text = event.text if hasattr(event, 'text') else str(event)
                if workflow and user_text:
                    workflow.add_transcript_entry("user", user_text)
                    logger.debug(f"📝 Captured user message via event: {user_text[:50]}...")
                    
                    # Try to detect activity/technique from user message
                    if metric_discovery:
                        detected_activity = metric_discovery._detect_activity_from_text(user_text, None)
                        if detected_activity and (not workflow.current_activity or workflow.current_activity == "unknown"):
                            workflow.current_activity = detected_activity
                            logger.info(f"✅ Detected activity from user message: {detected_activity}")
                    
                    if request_handler:
                        technique = request_handler._extract_technique(user_text)
                        if technique and (not workflow.current_technique or workflow.current_technique == "unknown"):
                            workflow.current_technique = technique
                            logger.info(f"✅ Detected technique from user message: {technique}")
        
        # Try to subscribe to agent speech transcription events if available
        if hasattr(agent, 'on_realtime_agent_speech_transcription'):
            @agent.on_realtime_agent_speech_transcription
            async def on_agent_transcript(event):
                """Capture all agent speech transcriptions"""
                agent_text = event.text if hasattr(event, 'text') else str(event)
                if workflow and agent_text:
                    workflow.add_transcript_entry("assistant", agent_text)
                    logger.debug(f"📝 Captured agent response via event: {agent_text[:50]}...")
    except Exception as e:
        logger.debug(f"Could not set up event handlers (may not be available): {e}")
        # Fall back to explicit tracking - user_request_handler and explicit tracking in simple_response calls
    
    with await agent.join(call):
        logger.info("✅ Agent joined call")
        
        # Wait for stream initialization
        await asyncio.sleep(3)
        
        # Note: Frame processing is handled by MetricsProcessor
        # The processor's add_pose_to_ndarray method will be called automatically
        # by the vision-agents framework for each video frame
        
        # Set up event handlers AFTER joining to capture ALL user messages
        # Subscribe to user speech transcription events using agent's subscribe method
        try:
            from vision_agents.core.llm.events import RealtimeUserSpeechTranscriptionEvent, RealtimeAgentSpeechTranscriptionEvent
            
            @agent.subscribe
            async def on_user_transcript(event: RealtimeUserSpeechTranscriptionEvent):
                """Capture ALL user speech transcriptions"""
                user_text = event.text if hasattr(event, 'text') else str(event)
                if workflow and user_text and user_text.strip():
                    workflow.add_transcript_entry("user", user_text)
                    logger.debug(f"📝 Captured user message via event: {user_text[:50]}...")
                    
                    # Check for frame capture control commands
                    # Frame capture is always active - no start/stop control needed
                    # Removed start/end control for ACL-focused analysis
                    
                    # Check for ACL analysis request and add to user_requests
                    if "acl" in user_text_lower or "acl tear" in user_text_lower or "acl risk" in user_text_lower or "acl insight" in user_text_lower or "acl based" in user_text_lower or "acot" in user_text_lower:
                        if not workflow.user_requests:
                            workflow.user_requests = []
                        # Add ACL request if not already present
                        acl_requests = [req for req in workflow.user_requests if "acl" in req.lower()]
                        if not acl_requests:
                            workflow.user_requests.append("acl tear risk")
                            logger.info("✅ Detected ACL analysis request from user message - added to user_requests")
                            print("✅ [ACL] ACL analysis request detected - will track ACL risk metrics")
                    
                    # Try to detect activity/technique from user message
                    if metric_discovery:
                        detected_activity = metric_discovery._detect_activity_from_text(user_text, None)
                        if detected_activity and (not workflow.current_activity or workflow.current_activity == "unknown"):
                            workflow.current_activity = detected_activity
                            logger.info(f"✅ Detected activity from user message: {detected_activity}")
                    
                    if request_handler:
                        technique = request_handler._extract_technique(user_text)
                        if technique and (not workflow.current_technique or workflow.current_technique == "unknown"):
                            workflow.current_technique = technique
                            logger.info(f"✅ Detected technique from user message: {technique}")
            
            @agent.subscribe
            async def on_agent_transcript(event: RealtimeAgentSpeechTranscriptionEvent):
                """Capture ALL agent speech transcriptions"""
                agent_text = event.text if hasattr(event, 'text') else str(event)
                if workflow and agent_text and agent_text.strip():
                    workflow.add_transcript_entry("assistant", agent_text)
                    logger.debug(f"📝 Captured agent response via event: {agent_text[:50]}...")
        except ImportError:
            logger.debug("Could not import event types - event handlers may not work")
        except Exception as e:
            logger.debug(f"Could not set up event handlers: {e}")
            # Fall back to explicit tracking
        
        # Store initial activity/technique in workflow if provided
        if workflow:
            # Always set activity to gymnastics for cvMLAgentNew
            workflow.current_activity = selected_activity or "gymnastics"
            if selected_technique:
                workflow.current_technique = selected_technique
            else:
                # Set to unknown - will be detected automatically from video
                workflow.current_technique = "unknown"
                logger.info("🤸 Technique set to 'unknown' - will be detected automatically from video frames")
        
        # Build greeting based on what's provided - CONCISE ACL FOCUS
        if not selected_activity or not selected_technique:
            greeting = """🤸 ACL Risk Analysis Agent

Analyzing gymnastics performance for ACL tear risk during landing phases.

**Focus**: Landing technique and ACL risk factors
- Knee valgus collapse detection
- Landing stiffness assessment
- Impact force analysis
- HIGH risk moment identification

Analysis is active. Start performing when ready."""
        else:
            # Activity and technique provided - show activity-specific greeting
            activity_greetings = {
                "gymnastics": """🤸 ACL Risk Analysis Agent

Analyzing gymnastics performance for ACL tear risk during landing phases.

**Focus**: Landing technique and ACL risk factors
- Technique: {selected_technique or "Detected automatically"}
- ACL risk metrics calculated during landing phases
- HIGH risk moments flagged automatically

Analysis is active. Start performing when ready.""",
                "fitness": """Hello! I'm your CV ML Agent for Fitness Exercise Analytics with Deep Research capabilities.

I can analyze:
- Range of motion
- Tempo control
- Spinal alignment
- Joint alignment
- Exercise-specific form metrics
- And much more based on your exercise!

**Current Setup:**
- Activity: Fitness
- Technique: {selected_technique}

**How to use:**
1. Specify what you want tracked or improved:
   - "I want to track my squat depth and knee alignment"
   - "I want to improve my deadlift form"
   - "Check my push-up form and range of motion"
2. I'll automatically:
   - Use deep research to discover relevant metrics
   - Select the best ML models for your needs
   - Extract all relevant metrics
   - Compare to exercise science standards
   - Provide comprehensive analysis

**If you don't specify, I'll use default fitness metrics:**
- Range of motion
- Tempo control
- Spinal alignment
- Joint alignment

**Frame Capture Control:**
- Frame capture is **INACTIVE** by default
- Say **"start"** to begin capturing and analyzing frames
- Say **"end"** or **"stop"** to stop capturing frames
- This allows you to control exactly when analysis begins and ends

Ready when you are!""",
                "posture": """Hello! I'm your CV ML Agent for Posture Analysis with Deep Research capabilities.

I can analyze:
- Rounded/arched back (kyphosis/hyperlordosis)
- Hunched shoulders
- Forward head posture
- Hyperextended knees
- Bowed knees (varus/valgus)
- Hip alignment
- And much more!

**Current Setup:**
- Activity: Posture
- Technique: {selected_technique}

**How to use:**
1. Specify what you want tracked or improved:
   - "Check my forward head posture and shoulder alignment"
   - "I want to improve my sitting posture"
   - "Analyze my spinal alignment"
2. I'll automatically:
   - Use deep research to discover relevant metrics
   - Select the best ML models for your needs
   - Extract all relevant metrics
   - Compare to biomechanical standards
   - Provide comprehensive analysis

**If you don't specify, I'll use default posture metrics:**
- Rounded back
- Arched back
- Hunched shoulders
- Forward head posture
- Hyperextended knees
- Bowed knees
- Hip alignment

**Frame Capture Control:**
- Frame capture is **INACTIVE** by default
- Say **"start"** to begin capturing and analyzing frames
- Say **"end"** or **"stop"** to stop capturing frames
- This allows you to control exactly when analysis begins and ends

Ready when you are!""",
                "yoga": """Hello! I'm your CV ML Agent for Yoga Form Analysis with Deep Research capabilities.

I can analyze:
- Joint stacking
- Spinal alignment
- Foundation stability
- Breath awareness
- Pose-specific alignment
- And much more!

**Current Setup:**
- Activity: Yoga
- Technique: {selected_technique}

**How to use:**
1. Specify what you want tracked or improved:
   - "I want to improve my Warrior I hip alignment"
   - "Check my Downward Dog spinal alignment"
   - "Track my Tree Pose balance and alignment"
2. I'll automatically:
   - Use deep research to discover relevant metrics
   - Select the best ML models for your needs
   - Extract all relevant metrics
   - Compare to traditional yoga alignment standards
   - Provide comprehensive analysis

**If you don't specify, I'll use default yoga metrics:**
- Joint stacking
- Spinal alignment
- Foundation stability
- Breath awareness

**Frame Capture Control:**
- Frame capture is **INACTIVE** by default
- Say **"start"** to begin capturing and analyzing frames
- Say **"end"** or **"stop"** to stop capturing frames
- This allows you to control exactly when analysis begins and ends

Ready when you are!"""
            }
            
            greeting = activity_greetings.get(selected_activity, activity_greetings["gymnastics"]).format(
                selected_technique=selected_technique
            )
        
        # Add greeting to transcript
        if workflow:
            workflow.add_transcript_entry("assistant", greeting)
        
        await agent.llm.simple_response(text=greeting)
        
        # Set up event handlers to capture transcript
        # Note: vision-agents may provide event handlers for user/agent messages
        # We'll also track via LLM response monitoring
        
        # Start workflow monitoring
        workflow_task = asyncio.create_task(
            monitor_workflow(agent, workflow)
        )
        
        # Start metric discovery monitoring (check LLM responses)
        discovery_task = asyncio.create_task(
            monitor_metric_discovery(agent)
        )
        
        # Start transcript monitoring
        transcript_task = asyncio.create_task(
            monitor_transcript(agent, workflow)
        )
        
        try:
            await agent.finish()
        finally:
            workflow_task.cancel()
            discovery_task.cancel()
            transcript_task.cancel()
            try:
                await workflow_task
                await discovery_task
                await transcript_task
            except asyncio.CancelledError:
                pass
            
            # Get final activity/technique from workflow (may have been updated during call)
            # Priority: workflow > initial kwargs > "unknown"
            if workflow:
                final_activity = (workflow.current_activity if hasattr(workflow, 'current_activity') and workflow.current_activity 
                                else selected_activity or "unknown")
                final_technique = (workflow.current_technique if hasattr(workflow, 'current_technique') and workflow.current_technique 
                                 else selected_technique or "unknown")
            else:
                final_activity = selected_activity or "unknown"
                final_technique = selected_technique or "unknown"
            
            logger.info(f"💾 Saving outputs with activity='{final_activity}', technique='{final_technique}', call_id='{call_id}'")
            
            # Save metrics JSON, overlay video, and transcript when call ends
            if workflow:
                await save_call_outputs(workflow, final_activity, final_technique, call_id)


async def monitor_metric_discovery(agent):
    """
    Monitor for metric discovery requests in LLM responses.
    Periodically checks if LLM has detected user metric requests.
    """
    global request_handler
    
    await asyncio.sleep(5)  # Wait for initial setup
    
    while True:
        try:
            await asyncio.sleep(3)  # Check every 3 seconds
            
            # In vision-agents, we can't directly access LLM responses
            # Instead, we rely on the LLM responding with METRIC_DISCOVERY markers
            # This is a placeholder - actual implementation would hook into LLM response stream
            # For now, the LLM will respond with markers when it detects requests
            
        except asyncio.CancelledError:
            logger.info("Metric discovery monitoring cancelled")
            break
        except Exception as e:
            logger.error(f"Error in metric discovery monitoring: {e}", exc_info=True)
            await asyncio.sleep(5)


async def monitor_workflow(agent, workflow: AgenticMLWorkflow):
    """Monitor and execute the agentic ML workflow"""
    global request_handler
    
    await asyncio.sleep(5)  # Wait for initial setup
    
    while True:
        try:
            await asyncio.sleep(2)  # Check every 2 seconds
            
            # Check if workflow has tasks to execute
            if workflow and workflow.has_pending_tasks():
                await workflow.execute_next_step(agent)
            
            # Also process frames if available (for metric calculation)
            if workflow and hasattr(workflow, 'process_frame_for_metrics'):
                # Try to get latest frame from workflow state
                latest_frame = workflow.workflow_state.get("latest_frame")
                if latest_frame is not None:
                    # Process frame to extract pose and calculate metrics
                    workflow.process_frame_for_metrics(
                        latest_frame,
                        workflow.workflow_state.get("latest_frame_timestamp")
                    )
                
        except asyncio.CancelledError:
            logger.info("Workflow monitoring cancelled")
            break
        except Exception as e:
            logger.error(f"Error in workflow monitoring: {e}", exc_info=True)
            await asyncio.sleep(5)


async def parse_and_handle_technique_research(agent, llm_response_text: str) -> bool:
    """
    Parse and handle technique research requests.
    Triggered when agent detects unknown technique.
    """
    global request_handler, workflow, metric_discovery, deep_search
    
    if not llm_response_text or not request_handler:
        return False
    
    # Check for TECHNIQUE_RESEARCH marker
    if "🔍 TECHNIQUE_RESEARCH:" in llm_response_text or "TECHNIQUE_RESEARCH:" in llm_response_text:
        # Extract technique name
        match = re.search(r"TECHNIQUE_RESEARCH:(.+)", llm_response_text)
        if match:
            technique_name = match.group(1).strip()
            logger.info(f"🔍 Detected technique research request: {technique_name}")
            
            try:
                # Extract activity if mentioned
                detected_activity = metric_discovery._detect_activity_from_text(llm_response_text, technique_name)
                
                # Research the technique using deep search
                if deep_search:
                    research_result = deep_search.research_technique(
                        technique=technique_name,
                        sport=detected_activity or "general",
                        user_requests=None
                    )
                    
                    logger.info(f"✅ Researched technique '{technique_name}': {len(research_result.get('key_metrics', []))} metrics found")
                    
                    # Discover metrics based on research
                    discovery = metric_discovery.discover_metrics_from_request(
                        user_text=f"analyze {technique_name}",
                        technique=technique_name,
                        activity=detected_activity,
                        research_findings=research_result
                    )
                    
                    discovered_metrics = discovery.get("metrics_to_track", [])
                    
                    # Update workflow with researched technique and metrics
                    if workflow:
                        await workflow.initialize_workflow(
                            technique=technique_name,
                            user_requests=discovered_metrics,
                            research_findings=research_result,
                            activity=detected_activity
                        )
                        logger.info(f"✅ Workflow updated with researched technique and metrics")
                    
                    # Respond to user
                    response = f"🔬 I've researched '{technique_name}' and discovered {len(discovered_metrics)} key metrics to track:\n\n"
                    for i, metric in enumerate(discovered_metrics, 1):
                        response += f"{i}. {metric}\n"
                    response += f"\n✅ I'll now track these metrics for '{technique_name}'!"
                    
                    # Add to transcript (already handled by wrapper if available)
                    if workflow and hasattr(workflow, 'add_transcript_entry'):
                        workflow.add_transcript_entry("assistant", response)
                    
                    await agent.llm.simple_response(text=response)
                    
                    return True
                    
            except Exception as e:
                logger.error(f"❌ Error researching technique: {e}", exc_info=True)
                return False
    
    return False


async def parse_and_handle_metric_discovery(agent, llm_response_text: str) -> bool:
    """
    Parse LLM response for metric discovery requests.
    The LLM is instructed to respond with METRIC_DISCOVERY marker when it detects user requests.
    
    Args:
        agent: The agent instance
        llm_response_text: Text from LLM response
    
    Returns:
        True if metric discovery was triggered, False otherwise
    """
    global request_handler, workflow, metric_discovery
    
    if not llm_response_text or not request_handler:
        return False
    
    # Check for TECHNIQUE_RESEARCH marker first (unknown techniques)
    if "🔍 TECHNIQUE_RESEARCH:" in llm_response_text or "TECHNIQUE_RESEARCH:" in llm_response_text:
        handled = await parse_and_handle_technique_research(agent, llm_response_text)
        if handled:
            return True
    
    # Check for METRIC_DISCOVERY marker
    if "🔍 METRIC_DISCOVERY:" in llm_response_text or "METRIC_DISCOVERY:" in llm_response_text:
        # Extract user request
        match = re.search(r"METRIC_DISCOVERY:(.+)", llm_response_text)
        if match:
            user_request = match.group(1).strip()
            logger.info(f"🔍 Detected metric discovery request: {user_request}")
            
            try:
                # Extract activity and technique if mentioned
                technique = request_handler._extract_technique(user_request)
                
                # Try to detect activity from user request
                detected_activity = metric_discovery._detect_activity_from_text(user_request, technique)
                
                # Discover metrics using deep research
                discovery = metric_discovery.discover_metrics_from_request(
                    user_text=user_request,
                    technique=technique,
                    activity=detected_activity
                )
                
                discovered_metrics = discovery.get("metrics_to_track", [])
                from_research = discovery.get("discovered_from_research", False)
                
                logger.info(f"✅ Discovered {len(discovered_metrics)} metrics")
                if from_research:
                    logger.info("   📚 Metrics discovered from deep research")
                else:
                    logger.info("   ℹ️  Using base metrics (research not available)")
                
                # Update workflow with discovered metrics
                if workflow and technique:
                    await workflow.initialize_workflow(
                        technique=technique,
                        user_requests=discovered_metrics
                    )
                    logger.info(f"✅ Workflow updated with discovered metrics")
                
                # Respond to user
                response = request_handler._format_discovery_response(discovery)
                
                # Add to transcript
                if workflow and hasattr(workflow, 'add_transcript_entry'):
                    workflow.add_transcript_entry("assistant", response)
                
                await agent.llm.simple_response(text=response)
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Error in metric discovery: {e}", exc_info=True)
                return False
    
    return False


async def monitor_transcript(agent, workflow: AgenticMLWorkflow):
    """
    Monitor and capture transcript entries from agent/user interactions.
    Attempts to hook into vision-agents event system for automatic capture.
    Falls back to explicit tracking if events not available.
    """
    global request_handler
    
    await asyncio.sleep(5)  # Wait for initial setup
    
    # Note: Event handlers are set up in join_call() if available
    # This monitoring task is a fallback/placeholder
    # Actual capture happens via:
    # 1. Event handlers (if available)
    # 2. Explicit tracking in user_request_handler
    # 3. Explicit tracking when sending agent responses
    
    while True:
        try:
            await asyncio.sleep(1)  # Check frequently
            # Event handlers do the actual work
        except asyncio.CancelledError:
            logger.info("Transcript monitoring cancelled")
            break
        except Exception as e:
            logger.debug(f"Error in transcript monitoring: {e}")
            await asyncio.sleep(5)


async def save_call_outputs(
    workflow: AgenticMLWorkflow,
    activity: Optional[str],
    technique: Optional[str],
    call_id: Optional[str] = None
):
    """
    Save metrics JSON, overlay video, and transcript when stream call ends.
    
    Args:
        workflow: The workflow instance with stored data
        activity: Activity name (gymnastics, fitness, posture, yoga)
        technique: Technique name
        call_id: Call/Video ID for filename
    """
    try:
        logger.info("💾 Saving call outputs...")
        
        # Create output directory
        output_dir = Path("stream_output")
        output_dir.mkdir(exist_ok=True)
        
        # Get activity/technique from workflow if available (may have been updated during call)
        # Priority: workflow > function parameter > "unknown"
        if workflow:
            final_activity = (workflow.current_activity if hasattr(workflow, 'current_activity') and workflow.current_activity 
                            else activity or "unknown")
            final_technique = (workflow.current_technique if hasattr(workflow, 'current_technique') and workflow.current_technique 
                             else technique or "unknown")
        else:
            final_activity = activity or "unknown"
            final_technique = technique or "unknown"
        
        # Generate filename with call_id if available, otherwise timestamp
        if call_id:
            # Use call_id as primary identifier
            activity_str = final_activity or "unknown"
            technique_str = final_technique or "unknown"
            base_name = f"{activity_str}_{technique_str}_{call_id}"
        else:
            # Fallback to timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            activity_str = final_activity or "unknown"
            technique_str = final_technique or "unknown"
            base_name = f"{activity_str}_{technique_str}_{timestamp}"
        
        # Initialize Cloudflare Stream URL (will be populated if video is uploaded)
        cloudflare_stream_url = None
        cloudflare_video_info = None
        
        # 1. Save metrics JSON with insights
        frame_metrics = workflow.workflow_state.get("frame_metrics", [])
        
        metrics_data = {
            # Cloudflare Stream URL (added early as requested)
            "cloudflare_stream_url": cloudflare_stream_url,
            "cloudflare_video_info": cloudflare_video_info,
            # Core identifiers
            "activity": activity_str,
            "technique": technique_str,
            "timestamp": datetime.utcnow().isoformat(),
            "call_id": call_id,
            # Workflow data
            "workflow_summary": workflow.get_workflow_summary(),
            "metrics": workflow.workflow_state.get("metrics", {}),
            "analysis": workflow.workflow_state.get("analysis", {}),
            "standards_comparison": workflow.workflow_state.get("analysis", {}).get("comparison", {}),
            "insights": _generate_insights(workflow),
            "acl_flagged_timesteps": workflow.workflow_state.get("acl_flagged_timesteps", []),
            "landing_phases": workflow.workflow_state.get("landing_phases", []),  # Detected landing phases
            "frame_metrics": frame_metrics,  # Frame-by-frame metrics for detailed analysis
            "frame_metrics_count": len(frame_metrics),
            "user_requests": workflow.user_requests,
            "selected_models": workflow.workflow_state.get("selected_models", {}),
            "workflow_sequence": workflow.workflow_sequence,
            "completed_steps": len(workflow.completed_steps),
            "total_steps": len(workflow.workflow_sequence)
        }
        
        metrics_file = output_dir / f"{base_name}_metrics.json"
        # Note: metrics_data will be updated later with Cloudflare URL if video is uploaded
        # We'll write it multiple times as we get more data (video info, Cloudflare URL, etc.)
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.info(f"✅ Metrics JSON saved to: {metrics_file}")
        
        # 2. Extract video clips for high-risk ACL frames and generate coach-friendly report
        acl_flagged_timesteps = workflow.workflow_state.get("acl_flagged_timesteps", [])
        user_requests = workflow.user_requests or []
        
        # Check transcript for ACL requests (user might have said it in conversation)
        transcript = workflow.transcript if hasattr(workflow, 'transcript') else workflow.workflow_state.get("transcript", [])
        transcript_text = " ".join([entry.get("content", "") for entry in transcript if entry.get("role") == "user"]).lower()
        
        acl_analysis_requested = (
            any("acl" in req.lower() or "acl tear" in req.lower() or "acl risk" in req.lower() for req in user_requests) or
            "acl" in transcript_text or "acl tear" in transcript_text or "acl risk" in transcript_text or
            "acl insight" in transcript_text or "acl based" in transcript_text
        )
        
        if acl_analysis_requested:
            logger.info(f"🔍 ACL analysis was requested - checking for high-risk moments and stored frames...")
            print(f"🔍 [ACL Clips] ACL analysis was requested by user")
            
            if not acl_flagged_timesteps:
                logger.info("ℹ️  No HIGH risk ACL moments detected (score >= 0.7) - no clips to extract")
                print(f"ℹ️  [ACL Clips] No HIGH risk moments detected (all risk scores were < 0.7)")
                print(f"   This is normal - clips are only extracted for HIGH risk moments (score >= 0.7)")
            else:
                logger.info(f"🎬 Extracting video clips for {len(acl_flagged_timesteps)} high-risk ACL frames...")
                print(f"🎬 [ACL Clips] Extracting {len(acl_flagged_timesteps)} video clips for high-risk ACL frames...")
                
                try:
                    from video_clip_extractor import VideoClipExtractor
                    clip_extractor = workflow.workflow_state.get("video_clip_extractor")
                    
                    if not clip_extractor:
                        logger.warning("⚠️  Video clip extractor not found in workflow state")
                        print("⚠️  [ACL Clips] Video clip extractor not available - frames may not have been stored during processing")
                    else:
                        # Check if frames were stored
                        stored_frames_count = len(clip_extractor.stored_frames) if hasattr(clip_extractor, 'stored_frames') else 0
                        logger.info(f"📹 Found {stored_frames_count} stored frames for clip extraction")
                        print(f"📹 [ACL Clips] Found {stored_frames_count} stored frames for clip extraction")
                        
                        if stored_frames_count == 0:
                            logger.warning("⚠️  No frames stored in clip extractor - clips cannot be generated")
                            print("⚠️  [ACL Clips] No frames stored - ensure frames are stored during processing")
                        else:
                            # Extract clips from stored frames
                            clips = clip_extractor.extract_clips_for_acl_risk(
                                acl_flagged_timesteps=acl_flagged_timesteps,
                                output_dir=output_dir,
                                base_name=base_name,
                                fps=30.0,
                                clip_duration_seconds=2.0,
                                frames_before=30,  # 1 second before
                                frames_after=30    # 1 second after
                            )
                            
                            if clips:
                                logger.info(f"✅ Extracted {len(clips)} ACL risk video clips")
                                print(f"✅ [ACL Clips] Successfully extracted {len(clips)} video clips:")
                                
                                # Add clip paths to flagged timesteps and log each clip
                                for i, clip in enumerate(clips):
                                    clip_path = clip.get("clip_path")
                                    if i < len(acl_flagged_timesteps):
                                        acl_flagged_timesteps[i]["video_clip_path"] = clip_path
                                    
                                    # Log each clip
                                    logger.info(f"  Clip {i+1}: {Path(clip_path).name} (Frame {clip.get('flagged_frame_number', 'N/A')}, Score: {clip.get('risk_score', 0.0):.2f})")
                                    print(f"  📹 Clip {i+1}: {Path(clip_path).name}")
                                    print(f"     Frame: {clip.get('flagged_frame_number', 'N/A')}, Risk Score: {clip.get('risk_score', 0.0):.2f}, Duration: {clip.get('clip_duration', 0.0):.2f}s")
                                
                                # Add clip info to metrics data
                                metrics_data["acl_risk_clips"] = clips
                                
                                # Update metrics file with clip info
                                with open(metrics_file, 'w') as f:
                                    json.dump(metrics_data, f, indent=2, default=str)
                                
                                logger.info(f"✅ ACL risk clips saved and metadata updated in metrics JSON")
                                print(f"✅ [ACL Clips] All clips saved to: {output_dir}/")
                            else:
                                logger.warning("⚠️  No clips were extracted (may be due to frame mismatch)")
                                print("⚠️  [ACL Clips] No clips extracted - check frame numbers match stored frames")
                except Exception as e:
                    logger.error(f"❌ Error extracting video clips: {e}", exc_info=True)
                    print(f"❌ [ACL Clips] Error extracting video clips: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
        
        # Even if no HIGH risk moments, check if frames were stored for future analysis
        if acl_analysis_requested:
            clip_extractor = workflow.workflow_state.get("video_clip_extractor")
            if clip_extractor:
                stored_frames_count = len(clip_extractor.stored_frames) if hasattr(clip_extractor, 'stored_frames') else 0
                if stored_frames_count > 0:
                    logger.info(f"ℹ️  {stored_frames_count} frames were stored during processing (available for future analysis)")
                    print(f"ℹ️  [ACL Clips] {stored_frames_count} frames stored during processing")
                else:
                    logger.warning("⚠️  No frames were stored - ensure frame capture was active and ACL analysis was requested")
                    print(f"⚠️  [ACL Clips] No frames stored - did you say 'start' to begin frame capture?")
            
            # Generate coach-friendly ACL risk report (even if no HIGH risk moments)
            try:
                from coach_acl_report import CoachACLReportGenerator
                report_generator = CoachACLReportGenerator()
                
                landing_phases = workflow.workflow_state.get("landing_phases", [])
                metrics = workflow.workflow_state.get("metrics", {})
                
                logger.info("📋 Generating coach-friendly ACL risk report...")
                report = report_generator.generate_report(
                    acl_flagged_timesteps=acl_flagged_timesteps,
                    landing_phases=landing_phases,
                    metrics=metrics,
                    activity=activity_str,
                    technique=technique_str,
                    athlete_name=None,  # Can be extracted from call metadata if available
                    session_date=datetime.utcnow().strftime("%Y-%m-%d")
                )
                
                # Save JSON report
                report_file = report_generator.save_report(report, output_dir, base_name)
                logger.info(f"✅ Coach-friendly ACL risk report (JSON) saved to: {report_file}")
                print(f"✅ [ACL Report] Coach-friendly report saved: {Path(report_file).name}")
                
                # Save text report
                text_file = report_generator.save_text_report(report, output_dir, base_name)
                logger.info(f"✅ Coach-friendly ACL risk report (TEXT) saved to: {text_file}")
                print(f"✅ [ACL Report] Text report saved: {Path(text_file).name}")
                
                # Add report paths to metrics data
                metrics_data["acl_risk_report_json"] = str(report_file)
                metrics_data["acl_risk_report_text"] = str(text_file)
                
                # Update metrics file with report paths
                with open(metrics_file, 'w') as f:
                    json.dump(metrics_data, f, indent=2, default=str)
                    
            except Exception as e:
                logger.warning(f"⚠️  Could not generate coach-friendly ACL risk report: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            
            # Even if no HIGH risk moments, check if frames were stored for future analysis
            clip_extractor = workflow.workflow_state.get("video_clip_extractor")
            if clip_extractor:
                stored_frames_count = len(clip_extractor.stored_frames) if hasattr(clip_extractor, 'stored_frames') else 0
                if stored_frames_count > 0:
                    logger.info(f"ℹ️  {stored_frames_count} frames were stored during processing (available for future analysis)")
                    print(f"ℹ️  [ACL Clips] {stored_frames_count} frames stored during processing")
                else:
                    logger.warning("⚠️  No frames were stored - ensure frame capture was active and ACL analysis was requested")
                    print(f"⚠️  [ACL Clips] No frames stored - did you say 'start' to begin frame capture?")
        
        # 3. Save captured video segment from stored frames (if available)
        clip_extractor = workflow.workflow_state.get("video_clip_extractor")
        if clip_extractor and hasattr(clip_extractor, 'stored_frames'):
            stored_frames_count = len(clip_extractor.stored_frames)
            if stored_frames_count > 0:
                try:
                    from video_clip_extractor import VideoClipExtractor
                    captured_video_file = output_dir / f"{base_name}_captured_segment.mp4"
                    
                    logger.info(f"🎬 Saving captured video segment ({stored_frames_count} frames)...")
                    print(f"🎬 [Video] Saving captured video segment ({stored_frames_count} frames)...")
                    
                    video_info = clip_extractor.save_all_frames_as_video(
                        output_path=captured_video_file,
                        fps=30.0,
                        add_timestamps=True,
                        add_frame_numbers=True
                    )
                    
                    if video_info:
                        logger.info(f"✅ Captured video segment saved to: {captured_video_file}")
                        print(f"✅ [Video] Captured video saved: {Path(captured_video_file).name}")
                        print(f"   Duration: {video_info.get('duration_seconds', 0.0):.2f}s, Frames: {video_info.get('frame_count', 0)}")
                        
                        # Upload to Cloudflare Stream
                        try:
                            from cloudflare_stream import CloudflareStreamUploader
                            uploader = CloudflareStreamUploader()
                            
                            metadata = {
                                "activity": activity_str,
                                "technique": technique_str,
                                "call_id": call_id,
                                "timestamp": datetime.utcnow().isoformat(),
                                "filename": captured_video_file.name
                            }
                            
                            cloudflare_result = uploader.upload_video(str(captured_video_file), metadata)
                            if cloudflare_result:
                                cloudflare_stream_url = cloudflare_result.get("stream_url")
                                cloudflare_video_info = cloudflare_result
                                
                                logger.info(f"✅ Video uploaded to Cloudflare Stream: {cloudflare_stream_url}")
                                print(f"✅ [Cloudflare] Video uploaded: {cloudflare_result.get('uid', 'N/A')}")
                                
                                # Update metrics data with Cloudflare info
                                metrics_data["cloudflare_stream_url"] = cloudflare_stream_url
                                metrics_data["cloudflare_video_info"] = cloudflare_video_info
                            else:
                                logger.warning("⚠️  Failed to upload video to Cloudflare Stream")
                                print("⚠️  [Cloudflare] Upload failed - check credentials")
                        except Exception as e:
                            logger.warning(f"⚠️  Error uploading to Cloudflare Stream: {e}")
                            print(f"⚠️  [Cloudflare] Error: {e}")
                        
                        # Add video info to metrics data
                        metrics_data["captured_video_segment"] = video_info
                        
                        # Update metrics file with video info (including Cloudflare URL)
                        with open(metrics_file, 'w') as f:
                            json.dump(metrics_data, f, indent=2, default=str)
                    else:
                        logger.warning("⚠️  Failed to save captured video segment")
                        print("⚠️  [Video] Failed to save captured video segment")
                except Exception as e:
                    logger.error(f"❌ Error saving captured video segment: {e}", exc_info=True)
                    print(f"❌ [Video] Error saving captured video segment: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            else:
                logger.info("ℹ️  No frames stored in clip extractor - skipping captured video save")
                print("ℹ️  [Video] No frames stored - captured video segment not saved")
        else:
            logger.info("ℹ️  Video clip extractor not found - skipping captured video save")
            print("ℹ️  [Video] Video clip extractor not found - captured video segment not saved")
        
        # 4. Save overlay video if frames are available (legacy support)
        frames = workflow.workflow_state.get("frames", [])
        if frames:
            video_file = output_dir / f"{base_name}_overlay.mp4"
            
            # Get video dimensions from first frame
            if frames and len(frames) > 0:
                first_frame = frames[0].get("frame")
                if first_frame is not None and isinstance(first_frame, np.ndarray):
                    height, width = first_frame.shape[:2]
                    fps = 30  # Default FPS
                    
                    # Try to get FPS from frame data
                    if frames[0].get("fps"):
                        fps = frames[0]["fps"]
                    
                    # Create video writer
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        str(video_file),
                        fourcc,
                        fps,
                        (width, height)
                    )
                    
                    # Write all frames
                    for frame_data in frames:
                        frame = frame_data.get("frame")
                        if frame is not None and isinstance(frame, np.ndarray):
                            video_writer.write(frame)
                    
                    video_writer.release()
                    logger.info(f"✅ Overlay video saved to: {video_file}")
                else:
                    logger.warning("⚠️  Frames not in correct format, skipping video save")
            else:
                logger.warning("⚠️  No valid frames to save")
        else:
            logger.info("ℹ️  No frames stored, skipping video save")
        
        # 4. Save transcript as text file
        transcript = workflow.transcript if hasattr(workflow, 'transcript') else workflow.workflow_state.get("transcript", [])
        transcript_file = None
        transcript_text = None
        if transcript:
            transcript_file = output_dir / f"{base_name}_transcript.txt"
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(f"Call Transcript\n")
                f.write(f"{'='*60}\n")
                f.write(f"Call ID: {call_id or 'N/A'}\n")
                f.write(f"Activity: {activity_str}\n")
                f.write(f"Technique: {technique_str}\n")
                f.write(f"Timestamp: {datetime.utcnow().isoformat()}\n")
                f.write(f"{'='*60}\n\n")
                
                for entry in transcript:
                    role = entry.get("role", "unknown")
                    content = entry.get("content", "")
                    timestamp = entry.get("timestamp", "")
                    
                    # Format entry
                    role_label = "USER" if role == "user" else "AGENT"
                    f.write(f"[{timestamp}] {role_label}:\n")
                    f.write(f"{content}\n")
                    f.write(f"\n{'-'*60}\n\n")
            
            logger.info(f"✅ Transcript saved to: {transcript_file}")
        else:
            logger.info("ℹ️  No transcript entries to save")
        
        # 5. Save to MongoDB (optional - gracefully handle if dependencies missing)
        try:
            # Check if pymongo/bson is available first
            try:
                import bson
            except ImportError:
                logger.info("ℹ️  MongoDB dependencies (pymongo/bson) not available - skipping MongoDB save")
                raise ImportError("MongoDB dependencies not available")
            
            # Import MongoDBService (add parent directory to path)
            import sys
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            
            from videoAgent.mongodb_service import MongoDBService
            
            mongodb = MongoDBService()
            if mongodb.connect():
                logger.info("💾 Saving session to MongoDB...")
                
                # Read transcript text if file exists
                if transcript_file and transcript_file.exists():
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        transcript_text = f.read()
                else:
                    transcript_text = None
                
                # Helper function to convert numpy arrays and other non-serializable types to MongoDB-compatible types
                def sanitize_for_mongodb(obj):
                    """Recursively convert numpy arrays and other non-serializable types to MongoDB-compatible types"""
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.integer, np.floating)):
                        return obj.item()
                    elif isinstance(obj, dict):
                        return {k: sanitize_for_mongodb(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [sanitize_for_mongodb(item) for item in obj]
                    elif isinstance(obj, (datetime, date)):
                        return obj.isoformat()
                    else:
                        return obj
                
                # Prepare session document (similar to mongodb_session_ingestion.py)
                # Extract ACL risk data
                acl_flagged = metrics_data.get("acl_flagged_timesteps", [])
                metrics_dict = metrics_data.get("metrics", {})
                
                # Extract ACL risk metrics
                acl_risk_score = metrics_dict.get("acl_tear_risk_score", 0.0)
                acl_risk_level = metrics_dict.get("acl_risk_level", "MINIMAL")
                acl_max_valgus = metrics_dict.get("acl_max_valgus_angle", 0.0)
                
                # Count risk moments by level
                high_risk_count = len([ts for ts in acl_flagged if ts.get("risk_score", 0.0) >= 0.7])
                moderate_risk_count = len([ts for ts in acl_flagged if 0.4 <= ts.get("risk_score", 0.0) < 0.7])
                total_risk_count = len(acl_flagged)
                
                # Determine risk flags
                has_risk_acl = (
                    acl_risk_score >= 0.4 or
                    total_risk_count > 0 or
                    acl_risk_level in ["MODERATE", "HIGH"]
                )
                has_high_risk_acl = (
                    acl_risk_score >= 0.7 or
                    high_risk_count > 0 or
                    acl_risk_level == "HIGH"
                )
                
                # Prepare session document
                session_doc = {
                    "original_filename": base_name,  # Used for upsert lookup
                    "session_id": base_name,
                    "activity": activity_str,
                    "technique": technique_str,
                    "timestamp": metrics_data.get("timestamp", datetime.utcnow().isoformat()),
                    "call_id": call_id,
                    
                    # Metrics data (sanitize numpy arrays)
                    "metrics": sanitize_for_mongodb(metrics_dict),
                    "workflow_summary": sanitize_for_mongodb(metrics_data.get("workflow_summary", {})),
                    "analysis": sanitize_for_mongodb(metrics_data.get("analysis", {})),
                    "insights": sanitize_for_mongodb(metrics_data.get("insights", {})),
                    
                    # ACL risk data (sanitize numpy arrays)
                    "acl_flagged_timesteps": sanitize_for_mongodb(acl_flagged),
                    "acl_risk_score": sanitize_for_mongodb(acl_risk_score),
                    "acl_risk_level": acl_risk_level,
                    "acl_max_valgus_angle": sanitize_for_mongodb(acl_max_valgus),
                    "acl_high_risk_count": high_risk_count,
                    "acl_moderate_risk_count": moderate_risk_count,
                    "acl_total_risk_count": total_risk_count,
                    "has_high_risk_acl": has_high_risk_acl,
                    "has_risk_acl": has_risk_acl,
                    
                    # Landing phases (sanitize numpy arrays)
                    "landing_phases": sanitize_for_mongodb(metrics_data.get("landing_phases", [])),
                    
                    # Frame metrics count
                    "frame_metrics_count": metrics_data.get("frame_metrics_count", 0),
                    
                    # Transcript
                    "transcript": transcript_text,
                    "transcript_file": str(transcript_file) if transcript_file else None,
                    
                    # File paths
                    "metrics_file": str(metrics_file),
                    
                    # Cloudflare Stream info (sanitize to remove any numpy arrays)
                    "cloudflare_stream_url": cloudflare_stream_url,
                    "cloudflare_video_info": sanitize_for_mongodb(cloudflare_video_info) if cloudflare_video_info else None,
                    
                    # Metadata
                    "ingested_at": datetime.utcnow().isoformat(),
                    "ingestion_source": "cvmlagent_stream"
                }
                
                # Upsert session to MongoDB
                session_id = mongodb.upsert_session_by_filename(session_doc)
                if session_id:
                    logger.info(f"✅ Session saved to MongoDB: {session_id}")
                else:
                    logger.warning("⚠️  Failed to save session to MongoDB")
            else:
                logger.warning("⚠️  MongoDB connection failed - skipping MongoDB save")
        except ImportError as e:
            # MongoDB dependencies (pymongo/bson) not available - this is optional
            logger.info(f"ℹ️  MongoDB dependencies not available - skipping MongoDB save ({e})")
            # Don't fail the whole save operation if MongoDB dependencies are missing
        except Exception as e:
            logger.warning(f"⚠️  Error saving to MongoDB: {e}", exc_info=True)
            # Don't fail the whole save operation if MongoDB fails
        
        logger.info(f"📁 All outputs saved to: {output_dir}/")
        
    except Exception as e:
        logger.error(f"❌ Error saving call outputs: {e}", exc_info=True)


def _generate_insights(workflow: AgenticMLWorkflow) -> Dict[str, Any]:
    """
    Generate insights from workflow metrics and analysis, including root causes and recommendations.
    
    Args:
        workflow: The workflow instance
    
    Returns:
        Dictionary with insights including root_causes and structured recommendations
    """
    insights = {
        "summary": "",
        "key_findings": [],
        "root_causes": [],
        "recommendations": [],
        "performance_score": None
    }
    
    try:
        metrics = workflow.workflow_state.get("metrics", {})
        analysis = workflow.workflow_state.get("analysis", {})
        comparison = analysis.get("comparison", {})
        
        # Generate summary
        technique_name = workflow.current_technique or "unknown technique"
        if metrics:
            metric_count = len(metrics)
            insights["summary"] = f"Analyzed {metric_count} metrics for {technique_name}"
        else:
            insights["summary"] = f"Analysis for {technique_name}"
        
        # Extract gaps for key findings
        gaps = comparison.get("gaps", [])
        if gaps:
            insights["key_findings"] = [
                {
                    "metric": gap.get("metric", "unknown"),
                    "current": gap.get("current_value"),
                    "target": gap.get("target_value"),
                    "gap": gap.get("gap"),
                    "status": "needs_improvement"
                }
                for gap in gaps[:10]  # Top 10 gaps
            ]
        
        # Extract root causes from LLM insights or generate from gaps
        llm_insights = comparison.get("llm_insights", {})
        root_causes_from_llm = llm_insights.get("root_causes", [])
        
        if root_causes_from_llm:
            # Use LLM-generated root causes
            insights["root_causes"] = [
                {
                    "metric": rc.get("metric", "unknown"),
                    "root_cause": rc.get("root_cause", ""),
                    "biomechanical_explanation": rc.get("biomechanical_explanation", ""),
                    "correlation_with_other_metrics": rc.get("correlation_with_other_metrics", "")
                }
                for rc in root_causes_from_llm[:10]  # Top 10 root causes
            ]
        elif gaps:
            # Generate root causes from gaps if LLM insights not available
            # Map common metrics to likely root causes
            root_cause_mapping = {
                "acl_tear_risk": "Knee valgus collapse, insufficient landing flexion, or high impact forces increasing ACL injury risk",
                "acl_risk": "Knee valgus collapse, insufficient landing flexion, or high impact forces increasing ACL injury risk",
                "landing_knee_bend": "Insufficient knee extension during flight or early flexion before landing",
                "height_off_floor": "Insufficient drive from takeoff or improper technique execution",
                "knee_straightness": "Limited flexibility or muscle tightness preventing full extension",
                "impact_force": "Hard landing due to insufficient shock absorption or improper landing technique",
                "rotation": "Insufficient rotation due to timing issues or lack of momentum",
                "split_angle": "Limited hip flexibility preventing full split extension",
                "hip_alignment": "Weak core stability or improper technique causing misalignment",
                "shoulder_position": "Muscle imbalance or poor posture affecting shoulder alignment",
                "forward_head": "Weak neck muscles or poor ergonomic setup",
                "rounded_back": "Weak core muscles or poor spinal alignment",
            }
            
            insights["root_causes"] = []
            for gap in gaps[:10]:
                metric_name = gap.get("metric", "").lower()
                root_cause = "Biomechanical or technique issue requiring further analysis"
                
                # Find matching root cause from mapping
                for key, cause in root_cause_mapping.items():
                    if key in metric_name:
                        root_cause = cause
                        break
                
                insights["root_causes"].append({
                    "metric": gap.get("metric", "unknown"),
                    "root_cause": root_cause,
                    "current_value": gap.get("current_value"),
                    "target_value": gap.get("target_value"),
                    "gap": gap.get("gap")
                })
        
        # Extract and structure recommendations
        recommendations_from_llm = llm_insights.get("recommendations", [])
        
        if recommendations_from_llm:
            # Structure LLM recommendations
            insights["recommendations"] = [
                {
                    "priority": rec.get("priority", "medium"),
                    "metric": rec.get("metric", ""),
                    "what": rec.get("recommendation", ""),
                    "why": rec.get("reasoning", ""),
                    "how": rec.get("coaching_cue", ""),
                    "expected_impact": rec.get("expected_improvement", "")
                }
                for rec in recommendations_from_llm[:10]  # Top 10 recommendations
            ]
        elif comparison.get("recommendations"):
            # Use comparison recommendations if LLM recommendations not available
            recs = comparison["recommendations"]
            if isinstance(recs, list):
                # If it's a list of strings, structure them
                if recs and isinstance(recs[0], str):
                    insights["recommendations"] = [
                        {
                            "priority": "medium",
                            "what": rec,
                            "why": "",
                            "how": "",
                            "expected_impact": ""
                        }
                        for rec in recs[:10]
                    ]
                else:
                    # If already structured, use as-is
                    insights["recommendations"] = recs[:10]
        
        # Performance score (if available)
        if comparison.get("all_met") is not None:
            if comparison["all_met"]:
                insights["performance_score"] = 1.0
            else:
                # Calculate score based on gaps
                total_gaps = len(gaps)
                if total_gaps == 0:
                    insights["performance_score"] = 1.0
                elif total_gaps <= 2:
                    insights["performance_score"] = 0.8
                elif total_gaps <= 5:
                    insights["performance_score"] = 0.6
                else:
                    insights["performance_score"] = 0.4
        
        # Add historical feedback if available
        if comparison.get("feedback_with_history"):
            insights["historical_feedback"] = comparison["feedback_with_history"]
        
        # Add technique errors if available
        if llm_insights.get("technique_errors"):
            insights["technique_errors"] = llm_insights["technique_errors"]
        
        # Add ACL flagged timesteps from workflow state (ONLY if user requested ACL analysis)
        user_requests_text = " ".join(workflow.user_requests).lower() if workflow.user_requests else ""
        acl_requested = (
            "acl" in user_requests_text or 
            "acl tear" in user_requests_text or
            "acl risk" in user_requests_text or
            "acl injury" in user_requests_text or
            "anterior cruciate" in user_requests_text
        )
        
        acl_flagged = workflow.workflow_state.get("acl_flagged_timesteps", [])
        if acl_requested and acl_flagged:
            # Only include ACL data if user requested it AND HIGH risk was detected
            insights["acl_flagged_timesteps"] = acl_flagged
            insights["acl_risk_detected"] = True
            insights["acl_risk_count"] = len(acl_flagged)
            
            # Find highest risk timestep
            if acl_flagged:
                highest_risk = max(acl_flagged, key=lambda x: x.get("risk_score", 0.0))
                insights["acl_highest_risk_timestep"] = highest_risk
        elif acl_requested:
            # User requested ACL analysis but no HIGH risk detected
            insights["acl_risk_detected"] = False
            insights["acl_risk_count"] = 0
            insights["acl_analysis_requested"] = True
        else:
            # User did not request ACL analysis - do not include any ACL data
            insights["acl_analysis_requested"] = False
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}", exc_info=True)
    
    return insights


if __name__ == "__main__":
    if VISION_AGENTS_AVAILABLE:
        cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
    else:
        logger.info("Running in standalone mode - use workflow directly")
        # Can run workflow without vision-agents for testing
        logger.info("To use with vision-agents, install: pip install vision-agents[getstream,openai,ultralytics,gemini]")























