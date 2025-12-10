"""
Metrics Collection Utilities

Reusable metrics collection for tracking node/component execution.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import defaultdict


class MetricsCollector:
    """
    Collect metrics from nodes/components.
    
    Supports both simple recording and workflow-level tracking.
    """
    
    def __init__(self, name: str = "workflow"):
        """
        Initialize metrics collector.
        
        Args:
            name: Identifier for this collector (e.g., workflow name)
        """
        self.name = name
        self.metrics = defaultdict(list)
        self.node_metrics: Dict[str, Dict[str, Any]] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def start(self) -> None:
        """Mark start of execution."""
        self.start_time = datetime.now()
        self.node_metrics = {}
    
    def stop(self) -> None:
        """Mark end of execution."""
        self.end_time = datetime.now()
    
    def record(
        self,
        node_name: str,
        duration_ms: float,
        status: str,
        details: Optional[Dict] = None,
    ) -> None:
        """
        Record node execution (simple mode).
        
        Args:
            node_name: Node identifier
            duration_ms: Execution time in milliseconds
            status: "success" or "error"
            details: Optional additional details
        """
        self.metrics[node_name].append({
            "timestamp": time.time(),
            "duration_ms": duration_ms,
            "status": status,
            "details": details or {},
        })
    
    def record_node(
        self,
        node_name: str,
        duration_ms: float,
        status: str,
        input_keys: Optional[List[str]] = None,
        output_keys: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Record detailed node execution (workflow mode).
        
        Args:
            node_name: Node identifier
            duration_ms: Execution time in milliseconds
            status: Node status string
            input_keys: List of input state keys
            output_keys: List of output state keys
            warnings: List of warning messages
            error: Error message if failed
        """
        self.node_metrics[node_name] = {
            "name": node_name,
            "status": status,
            "duration_ms": duration_ms,
            "input_keys": input_keys or [],
            "output_keys": output_keys or [],
            "warnings": warnings or [],
            "error_message": error,
        }
        # Also record in simple mode
        self.record(node_name, duration_ms, status, {"error": error} if error else None)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics (simple mode).
        
        Returns:
            Dict with per-node statistics
        """
        summary = {}
        for node_name, executions in self.metrics.items():
            durations = [e["duration_ms"] for e in executions]
            summary[node_name] = {
                "count": len(executions),
                "avg_ms": sum(durations) / len(durations) if durations else 0,
                "min_ms": min(durations) if durations else 0,
                "max_ms": max(durations) if durations else 0,
                "success_rate": len([e for e in executions if e["status"] == "success"]) / len(executions) if executions else 0,
            }
        return summary
    
    def get_workflow_metrics(self) -> Dict[str, Any]:
        """
        Get detailed workflow metrics.
        
        Returns:
            Dict with workflow-level summary and per-node details
        """
        # Calculate total duration
        total_duration_ms = 0.0
        if self.start_time and self.end_time:
            total_duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        
        # Determine overall status
        statuses = [m.get("status") for m in self.node_metrics.values()]
        if "failed" in statuses or "error" in statuses:
            overall_status = "failed"
        elif "success" in statuses:
            overall_status = "success"
        else:
            overall_status = "unknown"
        
        # Collect all warnings
        all_warnings = []
        for metrics in self.node_metrics.values():
            all_warnings.extend(metrics.get("warnings", []))
        
        return {
            "workflow_name": self.name,
            "overall_status": overall_status,
            "total_duration_ms": total_duration_ms,
            "nodes_executed": len(self.node_metrics),
            "total_warnings": len(all_warnings),
            "nodes": self.node_metrics,
            "warnings": all_warnings,
        }

