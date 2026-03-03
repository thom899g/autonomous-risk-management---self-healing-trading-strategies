"""
Real-time risk monitoring system with multi-factor risk assessment
Architecture: Uses a composite risk score from multiple indicators with decaying memory
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from firebase_admin import firestore
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    HALTED = "halted"


@dataclass
class RiskIndicator:
    """Individual risk indicator with weight and threshold"""
    name: str
    current_value: float = 0.0
    threshold: float = 0.0
    weight: float = 1.0
    decay_factor: float = 0.95  # Memory decay for time-based indicators
    history: List[float] = field(default_factory=list)
    
    def update(self, new_value: float) -> None:
        """Update indicator with exponential decay memory"""
        if self.history:
            smoothed = (self.decay_factor * self.history[-1] + 
                       (1 - self.decay_factor) * new_value)
        else:
            smoothed = new_value
        self.current_value = smoothed
        self.history.append(smoothed)
        
        # Keep history manageable
        if len(self.history) > 1000:
            self.history = self.history[-1000:]


class RiskMonitor:
    """Main risk monitoring system with real-time assessment"""
    
    def __init__(self, 
                 strategy_id: str,
                 firestore_client: Optional[firestore.Client] = None,
                 update_interval: int = 30):
        """
        Initialize risk monitor for a specific trading strategy
        
        Args:
            strategy_id: Unique identifier for the trading strategy
            firestore_client: Firebase Firestore client (optional)
            update_interval: Monitoring interval in seconds
        """
        self.strategy_id = strategy_id
        self.db = firestore_client or firestore.client()
        self.update_interval = update_interval
        self.is_running = False
        
        # Initialize risk indicators
        self.indicators: Dict[str, RiskIndicator] = {
            "drawdown": RiskIndicator("drawdown", threshold=0.15, weight=1.5),
            "volatility": RiskIndicator("volatility", threshold=0.25, weight=1.2),
            "position_concentration": RiskIndicator("position_concentration", threshold=0.4, weight=1.3),
            "correlation_risk": RiskIndicator("correlation_risk", threshold=0.8, weight=1.1),
            "liquidity_risk": RiskIndicator("liquidity_risk", threshold=0.7, weight=1.0),
            "sentiment_risk": RiskIndicator("sentiment_risk", threshold=0.6, weight=0.9),
        }
        
        # Initialize state
        self.current_risk_level = RiskLevel.LOW
        self.risk_score = 0.0
        self.last_update = datetime.utcnow()
        self._setup_firestore_listeners()
        
        logger.info(f"Initialized RiskMonitor for strategy {strategy_id}")
    
    def _setup_firestore_listeners(self) -> None:
        """Set up Firestore listeners for real-time data"""
        try:
            # Listen for market data updates
            market_ref = self.db.collection("market_data").document(self.strategy_id)
            market_ref.on_snapshot(self._on_market_data_update)
            
            # Listen for position updates
            position_ref = self.db.collection("positions").document(self.strategy_id)
            position_ref.on_snapshot(self._on_position_update)
            
            logger.info(f"Firestore listeners established for {self.strategy_id}")
        except Exception as e:
            logger.error(f"Failed to setup Firestore listeners: {e}")
            raise
    
    def _on_market_data_update(self, snapshot, changes, read_time):
        """Handle market data updates from Firestore"""
        try:
            if snapshot.exists:
                data = snapshot.to_dict()
                self._process_market_data(data)
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
    
    def _on