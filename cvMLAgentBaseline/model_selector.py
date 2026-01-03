#!/usr/bin/env python3
"""
ML Model Selector with Reasoning
Selects appropriate ML models based on user requests, technique requirements, and available resources.
Provides reasoning on the sequence of steps and model selection.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of ML models available"""
    POSE_ESTIMATION = "pose_estimation"
    FACE_DETECTION = "face_detection"
    PERSON_DETECTION = "person_detection"
    WEIGHT_ESTIMATION = "weight_estimation"
    TECHNIQUE_CLASSIFICATION = "technique_classification"
    METRIC_EXTRACTION = "metric_extraction"


class ModelVariant(Enum):
    """Specific model variants"""
    # Pose estimation
    YOLO_POSE = "yolo_pose"
    HRNET = "hrnet"  # Primary method (from hrnet-metrics-server)
    OPENPOSE = "openpose"
    
    # Face detection
    OPENCV_DNN_FACE = "opencv_dnn_face"  # Optional for gymnastics
    
    # Person detection
    YOLO_PERSON = "yolo_person"
    HRNET_PERSON_DETECTOR = "hrnet_person_detector"  # Primary (from hrnet-metrics-server)
    
    # Weight estimation
    POSE_BASED_WEIGHT = "pose_based_weight"
    BODY_SHAPE_ESTIMATION = "body_shape_estimation"


class ModelReasoning:
    """Reasoning about model selection and workflow"""
    
    def __init__(self):
        self.reasoning_steps: List[Dict[str, Any]] = []
        self.selected_models: Dict[ModelType, ModelVariant] = {}
        self.workflow_sequence: List[Dict[str, Any]] = []
    
    def add_reasoning_step(self, step: str, reasoning: str, decision: Any):
        """Add a reasoning step"""
        self.reasoning_steps.append({
            "step": step,
            "reasoning": reasoning,
            "decision": decision,
            "timestamp": str(logging.Formatter().formatTime(logging.LogRecord(
                name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
            )))
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert reasoning to dictionary"""
        return {
            "reasoning_steps": self.reasoning_steps,
            "selected_models": {k.value: v.value for k, v in self.selected_models.items()},
            "workflow_sequence": self.workflow_sequence
        }


class MLModelSelector:
    """
    Selects ML models based on requirements with reasoning.
    """
    
    def __init__(self):
        self.available_models = self._initialize_available_models()
        self.model_capabilities = self._initialize_capabilities()
        self.reasoning = ModelReasoning()
    
    def _initialize_available_models(self) -> Dict[ModelType, List[ModelVariant]]:
        """Initialize available model variants"""
        return {
            ModelType.POSE_ESTIMATION: [
                ModelVariant.HRNET,  # Primary (from hrnet-metrics-server)
                ModelVariant.YOLO_POSE,  # Fallback
            ],
            ModelType.FACE_DETECTION: [
                ModelVariant.OPENCV_DNN_FACE,  # Optional
            ],
            ModelType.PERSON_DETECTION: [
                ModelVariant.HRNET_PERSON_DETECTOR,  # Primary (from hrnet-metrics-server)
                ModelVariant.YOLO_PERSON,  # Fallback
            ],
            ModelType.WEIGHT_ESTIMATION: [
                ModelVariant.POSE_BASED_WEIGHT,
                ModelVariant.BODY_SHAPE_ESTIMATION,
            ],
        }
    
    def _initialize_capabilities(self) -> Dict[str, List[ModelVariant]]:
        """Map capabilities to model variants"""
        return {
            # Pose-related capabilities (HRNet preferred for detailed analysis)
            "height_off_floor": [ModelVariant.HRNET, ModelVariant.YOLO_POSE],
            "knee_angles": [ModelVariant.HRNET, ModelVariant.YOLO_POSE],
            "landing_bend_angles": [ModelVariant.HRNET, ModelVariant.YOLO_POSE],
            "knee_straightness": [ModelVariant.HRNET, ModelVariant.YOLO_POSE],
            "stiffness": [ModelVariant.HRNET],  # HRNet provides more detailed joint angles
            "impact_force": [ModelVariant.HRNET, ModelVariant.YOLO_POSE],  # From velocity calculation
            
            # Face-related (optional)
            "face_detection": [ModelVariant.OPENCV_DNN_FACE],
            
            # Person-related (HRNet PersonDetector preferred)
            "person_detection": [ModelVariant.HRNET_PERSON_DETECTOR, ModelVariant.YOLO_PERSON],
            "person_tracking": [ModelVariant.HRNET_PERSON_DETECTOR, ModelVariant.YOLO_PERSON],
            
            # Weight-related
            "weight_estimation": [ModelVariant.POSE_BASED_WEIGHT, ModelVariant.BODY_SHAPE_ESTIMATION],
            "body_shape": [ModelVariant.BODY_SHAPE_ESTIMATION],
        }
    
    def select_models(
        self,
        user_requests: List[str],
        technique: Optional[str] = None,
        available_resources: Optional[Dict[str, Any]] = None,
        use_llm_reasoning: bool = False,
        llm_instance = None,
        research_findings: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[ModelType, ModelVariant], ModelReasoning]:
        """
        Select appropriate models based on user requests with reasoning.
        
        Args:
            user_requests: List of requested metrics/analyses
            technique: Optional technique name
            available_resources: Available computational resources
            use_llm_reasoning: If True, use LLM for reasoning (default: False, uses rule-based)
            llm_instance: LLM instance for reasoning (if use_llm_reasoning=True)
            research_findings: Optional research findings from deep search
        
        Returns:
            Tuple of (selected_models, reasoning)
        """
        self.reasoning = ModelReasoning()
        
        # Use LLM reasoning if requested and available
        if use_llm_reasoning and llm_instance:
            try:
                from llm_reasoning_selector import LLMReasoningSelector
                llm_selector = LLMReasoningSelector(llm_instance=llm_instance)
                llm_reasoning = llm_selector.reason_about_model_selection(
                    user_requests=user_requests,
                    technique=technique or "unknown",
                    research_findings=research_findings
                )
                
                # Convert LLM reasoning to our format
                selected_models = {}
                for model_type_str, variant_str in llm_reasoning.get("selected_models", {}).items():
                    try:
                        model_type = ModelType[model_type_str.upper()]
                        variant = ModelVariant[variant_str.upper()]
                        selected_models[model_type] = variant
                    except (KeyError, AttributeError):
                        logger.warning(f"⚠️  Unknown model type/variant from LLM: {model_type_str}/{variant_str}")
                
                # Add LLM reasoning steps
                for model_type_str, reason in llm_reasoning.get("reasoning", {}).items():
                    self.reasoning.add_reasoning_step(
                        f"llm_reasoning_{model_type_str}",
                        reason,
                        {"model": model_type_str}
                    )
                
                # Use LLM execution sequence if provided
                if llm_reasoning.get("execution_sequence"):
                    self.reasoning.workflow_sequence = llm_reasoning["execution_sequence"]
                
                self.reasoning.selected_models = selected_models
                logger.info("✅ Used LLM reasoning for model selection")
                return selected_models, self.reasoning
                
            except Exception as e:
                logger.warning(f"⚠️  LLM reasoning failed, falling back to rule-based: {e}")
                # Fall through to rule-based selection
        
        # Rule-based selection (original logic)
        # Step 1: Analyze user requests
        required_capabilities = self._extract_required_capabilities(user_requests)
        self.reasoning.add_reasoning_step(
            "analyze_requests",
            f"User requested: {', '.join(user_requests)}",
            {"required_capabilities": required_capabilities}
        )
        
        # Step 2: Determine required model types
        required_model_types = self._determine_model_types(required_capabilities)
        self.reasoning.add_reasoning_step(
            "determine_model_types",
            f"Required model types: {[t.value for t in required_model_types]}",
            {"model_types": [t.value for t in required_model_types]}
        )
        
        # Step 3: Select specific model variants
        selected_models = {}
        for model_type in required_model_types:
            variant = self._select_model_variant(
                model_type,
                required_capabilities,
                available_resources
            )
            selected_models[model_type] = variant
            
            self.reasoning.add_reasoning_step(
                f"select_{model_type.value}",
                f"Selected {variant.value} for {model_type.value}",
                {"variant": variant.value}
            )
        
        self.reasoning.selected_models = selected_models
        
        # Step 4: Determine workflow sequence
        workflow_sequence = self._determine_workflow_sequence(selected_models, technique)
        self.reasoning.workflow_sequence = workflow_sequence
        self.reasoning.add_reasoning_step(
            "determine_workflow",
            f"Workflow sequence: {len(workflow_sequence)} steps",
            {"sequence": workflow_sequence}
        )
        
        return selected_models, self.reasoning
    
    def _extract_required_capabilities(self, user_requests: List[str]) -> List[str]:
        """Extract required capabilities from user requests"""
        capabilities = []
        request_text = " ".join(user_requests).lower()
        
        # Map keywords to capabilities
        keyword_mapping = {
            "height": "height_off_floor",
            "height off floor": "height_off_floor",
            "impact": "impact_force",
            "impact force": "impact_force",
            "landing": "landing_bend_angles",
            "landing bend": "landing_bend_angles",
            "bend": "landing_bend_angles",
            "knee": "knee_straightness",
            "knee straight": "knee_straightness",
            "stiffness": "stiffness",
            "stiff": "stiffness",
            "face": "face_detection",
            "weight": "weight_estimation",
            "body shape": "body_shape",
            "person": "person_detection",
            "track": "person_tracking",
        }
        
        for keyword, capability in keyword_mapping.items():
            if keyword in request_text:
                if capability not in capabilities:
                    capabilities.append(capability)
        
        # Default: if no specific requests, assume pose estimation
        if not capabilities:
            capabilities = ["height_off_floor", "knee_angles"]
        
        return capabilities
    
    def _determine_model_types(self, capabilities: List[str]) -> List[ModelType]:
        """Determine which model types are needed"""
        model_types = set()
        
        for capability in capabilities:
            if capability in self.model_capabilities:
                # Get model variants that support this capability
                variants = self.model_capabilities[capability]
                # Map variants to model types
                for variant in variants:
                    if variant in [ModelVariant.YOLO_POSE, ModelVariant.HRNET]:
                        model_types.add(ModelType.POSE_ESTIMATION)
                    elif variant in [ModelVariant.OPENCV_DNN_FACE]:
                        model_types.add(ModelType.FACE_DETECTION)
                    elif variant in [ModelVariant.YOLO_PERSON, ModelVariant.HRNET_PERSON_DETECTOR]:
                        model_types.add(ModelType.PERSON_DETECTION)
                    elif variant in [ModelVariant.POSE_BASED_WEIGHT, ModelVariant.BODY_SHAPE_ESTIMATION]:
                        model_types.add(ModelType.WEIGHT_ESTIMATION)
        
        # Always need pose estimation for gymnastics
        if not model_types:
            model_types.add(ModelType.POSE_ESTIMATION)
        
        return list(model_types)
    
    def _select_model_variant(
        self,
        model_type: ModelType,
        capabilities: List[str],
        available_resources: Optional[Dict[str, Any]] = None
    ) -> ModelVariant:
        """Select specific model variant"""
        available_variants = self.available_models.get(model_type, [])
        
        if not available_variants:
            # Default fallback
            if model_type == ModelType.POSE_ESTIMATION:
                return ModelVariant.YOLO_POSE
            elif model_type == ModelType.FACE_DETECTION:
                return ModelVariant.MEDIAPIPE_FACE
            elif model_type == ModelType.PERSON_DETECTION:
                return ModelVariant.YOLO_PERSON
            else:
                return available_variants[0] if available_variants else None
        
        # Select based on capabilities and resources
        if model_type == ModelType.POSE_ESTIMATION:
            # ALWAYS prefer HRNet for detailed analysis (stiffness, detailed angles, gymnastics analytics)
            # HRNet is primary method (from hrnet-metrics-server) - more accurate for pose overlays
            if ModelVariant.HRNET in available_variants:
                return ModelVariant.HRNET
            # Fallback to YOLO only if HRNet not available
            if ModelVariant.YOLO_POSE in available_variants:
                logger.warning("⚠️  HRNet not available - falling back to YOLO for pose estimation")
                return ModelVariant.YOLO_POSE
            # No fallback available
            return available_variants[0] if available_variants else None
        
        elif model_type == ModelType.FACE_DETECTION:
            # OpenCV DNN (face detection is optional for gymnastics)
            if ModelVariant.OPENCV_DNN_FACE in available_variants:
                return ModelVariant.OPENCV_DNN_FACE
            return available_variants[0] if available_variants else None
        
        elif model_type == ModelType.PERSON_DETECTION:
            # Prefer HRNet PersonDetector (from hrnet-metrics-server)
            if ModelVariant.HRNET_PERSON_DETECTOR in available_variants:
                return ModelVariant.HRNET_PERSON_DETECTOR
            # Fallback to YOLO
            if ModelVariant.YOLO_PERSON in available_variants:
                return ModelVariant.YOLO_PERSON
            return available_variants[0] if available_variants else None
        
        elif model_type == ModelType.WEIGHT_ESTIMATION:
            # Prefer pose-based for gymnastics (more accurate from pose)
            if ModelVariant.POSE_BASED_WEIGHT in available_variants:
                return ModelVariant.POSE_BASED_WEIGHT
            return ModelVariant.BODY_SHAPE_ESTIMATION
        
        # Default to first available
        return available_variants[0]
    
    def _determine_workflow_sequence(
        self,
        selected_models: Dict[ModelType, ModelVariant],
        technique: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Determine the sequence of processing steps"""
        sequence = []
        
        # Step 1: Person detection (if needed)
        if ModelType.PERSON_DETECTION in selected_models:
            sequence.append({
                "step": 1,
                "model_type": ModelType.PERSON_DETECTION.value,
                "model_variant": selected_models[ModelType.PERSON_DETECTION].value,
                "purpose": "Detect and locate athlete in frame",
                "output": "bounding_box"
            })
        
        # Step 2: Face detection (if needed, can run in parallel with pose)
        if ModelType.FACE_DETECTION in selected_models:
            sequence.append({
                "step": 2,
                "model_type": ModelType.FACE_DETECTION.value,
                "model_variant": selected_models[ModelType.FACE_DETECTION].value,
                "purpose": "Detect face for identification/expression",
                "output": "face_landmarks",
                "parallel": True  # Can run in parallel with pose
            })
        
        # Step 3: Pose estimation (core step)
        if ModelType.POSE_ESTIMATION in selected_models:
            sequence.append({
                "step": 3,
                "model_type": ModelType.POSE_ESTIMATION.value,
                "model_variant": selected_models[ModelType.POSE_ESTIMATION].value,
                "purpose": "Extract body keypoints for metric calculation",
                "output": "keypoints",
                "depends_on": [1] if ModelType.PERSON_DETECTION in selected_models else []
            })
        
        # Step 4: Weight estimation (depends on pose)
        if ModelType.WEIGHT_ESTIMATION in selected_models:
            sequence.append({
                "step": 4,
                "model_type": ModelType.WEIGHT_ESTIMATION.value,
                "model_variant": selected_models[ModelType.WEIGHT_ESTIMATION].value,
                "purpose": "Estimate body weight/shape from pose",
                "output": "weight_estimate",
                "depends_on": [3]  # Needs pose data
            })
        
        # Step 5: Metric extraction (depends on pose)
        sequence.append({
            "step": 5,
            "model_type": "metric_extraction",
            "purpose": "Calculate technique-specific metrics",
            "output": "metrics",
            "depends_on": [3],  # Needs pose data
            "technique": technique
        })
        
        # Step 6: FIG standards comparison
        sequence.append({
            "step": 6,
            "model_type": "fig_standards_comparison",
            "purpose": "Compare metrics to FIG standards",
            "output": "standards_compliance",
            "depends_on": [5],  # Needs metrics
            "technique": technique
        })
        
        return sequence
    
    def get_reasoning_summary(self) -> str:
        """Get human-readable reasoning summary"""
        if not self.reasoning.reasoning_steps:
            return "No reasoning available"
        
        summary = "**Model Selection Reasoning:**\n\n"
        
        for step in self.reasoning.reasoning_steps:
            summary += f"**{step['step']}**: {step['reasoning']}\n"
            if isinstance(step['decision'], dict):
                for key, value in step['decision'].items():
                    summary += f"  - {key}: {value}\n"
            summary += "\n"
        
        if self.reasoning.workflow_sequence:
            summary += "\n**Workflow Sequence:**\n"
            for step in self.reasoning.workflow_sequence:
                summary += f"{step['step']}. {step['purpose']} ({step['model_type']})\n"
                if step.get('depends_on'):
                    summary += f"   Depends on: {step['depends_on']}\n"
        
        return summary






















