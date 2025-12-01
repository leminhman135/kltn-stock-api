"""
Scheduler Service - Tự động hóa các tác vụ định kỳ

Hỗ trợ:
- UptimeRobot: Keep-alive API (ping mỗi 5-10 phút)
- Render Cron Jobs: Chạy task định kỳ
- APScheduler: Python-based scheduling

Hiện tại sử dụng:
- UptimeRobot để keep-alive (https://uptimerobot.com)
- Render Cron cho daily tasks
"""

import os
import sys
import logging
import asyncio
import requests
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import threading
from functools import wraps

# Optional APScheduler
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    BackgroundScheduler = None
    AsyncIOScheduler = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Trạng thái task"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskPriority(Enum):
    """Mức độ ưu tiên"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskResult:
    """Kết quả thực hiện task"""
    task_id: str
    task_name: str
    status: TaskStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    result: Any = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'task_id': self.task_id,
            'task_name': self.task_name,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': round(self.duration_seconds, 2),
            'error': self.error
        }


@dataclass
class ScheduledTask:
    """Định nghĩa một scheduled task"""
    task_id: str
    name: str
    description: str
    function: Callable
    schedule: str  # Cron expression or interval
    enabled: bool = True
    priority: TaskPriority = TaskPriority.MEDIUM
    timeout_seconds: int = 3600  # 1 hour default
    retry_count: int = 3
    last_run: Optional[datetime] = None
    last_result: Optional[TaskResult] = None
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0


class TaskRegistry:
    """Registry để quản lý các tasks"""
    
    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.results: List[TaskResult] = []
        self.max_results = 1000  # Keep last 1000 results
    
    def register(self, task: ScheduledTask):
        """Đăng ký task"""
        self.tasks[task.task_id] = task
        logger.info(f"Registered task: {task.name} ({task.task_id})")
    
    def unregister(self, task_id: str):
        """Hủy đăng ký task"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            logger.info(f"Unregistered task: {task_id}")
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Lấy task theo ID"""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[ScheduledTask]:
        """Lấy tất cả tasks"""
        return list(self.tasks.values())
    
    def add_result(self, result: TaskResult):
        """Lưu kết quả task"""
        self.results.append(result)
        
        # Update task stats
        if result.task_id in self.tasks:
            task = self.tasks[result.task_id]
            task.last_run = result.start_time
            task.last_result = result
            task.run_count += 1
            if result.status == TaskStatus.SUCCESS:
                task.success_count += 1
            else:
                task.failure_count += 1
        
        # Trim old results
        if len(self.results) > self.max_results:
            self.results = self.results[-self.max_results:]
    
    def get_results(self, task_id: str = None, limit: int = 100) -> List[TaskResult]:
        """Lấy kết quả tasks"""
        if task_id:
            filtered = [r for r in self.results if r.task_id == task_id]
        else:
            filtered = self.results
        return filtered[-limit:]


class SchedulerService:
    """
    Service chính để schedule và chạy tasks
    
    Supports:
    - APScheduler (nếu installed)
    - Manual trigger via API
    - UptimeRobot keep-alive
    """
    
    def __init__(self, use_apscheduler: bool = True):
        self.registry = TaskRegistry()
        self.scheduler = None
        self.is_running = False
        
        if use_apscheduler and APSCHEDULER_AVAILABLE:
            self.scheduler = BackgroundScheduler()
            logger.info("Using APScheduler for task scheduling")
        else:
            logger.info("APScheduler not available, using manual mode")
    
    def _generate_task_id(self, name: str) -> str:
        """Tạo unique task ID"""
        timestamp = datetime.now().isoformat()
        hash_input = f"{name}_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _execute_task(self, task: ScheduledTask) -> TaskResult:
        """Thực hiện một task"""
        result = TaskResult(
            task_id=task.task_id,
            task_name=task.name,
            status=TaskStatus.RUNNING,
            start_time=datetime.now()
        )
        
        if not task.enabled:
            result.status = TaskStatus.SKIPPED
            result.end_time = datetime.now()
            return result
        
        logger.info(f"Executing task: {task.name}")
        
        try:
            # Run the task function
            task_result = task.function()
            
            result.status = TaskStatus.SUCCESS
            result.result = task_result
            
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = str(e)
            logger.error(f"Task {task.name} failed: {str(e)}")
        
        finally:
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            self.registry.add_result(result)
        
        logger.info(f"Task {task.name} completed: {result.status.value} "
                   f"({result.duration_seconds:.2f}s)")
        
        return result
    
    def add_task(self, 
                 name: str,
                 function: Callable,
                 schedule: str,
                 description: str = "",
                 enabled: bool = True,
                 priority: TaskPriority = TaskPriority.MEDIUM) -> str:
        """
        Thêm task mới
        
        Args:
            name: Tên task
            function: Hàm cần thực hiện
            schedule: Cron expression (e.g., "0 7 * * 1-5" = 7AM weekdays)
                     hoặc interval (e.g., "interval:hours=1")
            description: Mô tả
            enabled: Có enable không
            priority: Mức độ ưu tiên
        
        Returns:
            task_id
        """
        task_id = self._generate_task_id(name)
        
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            description=description,
            function=function,
            schedule=schedule,
            enabled=enabled,
            priority=priority
        )
        
        self.registry.register(task)
        
        # Add to APScheduler if available
        if self.scheduler and self.is_running:
            self._add_to_scheduler(task)
        
        return task_id
    
    def _add_to_scheduler(self, task: ScheduledTask):
        """Thêm task vào APScheduler"""
        if not self.scheduler or not APSCHEDULER_AVAILABLE:
            return
        
        try:
            if task.schedule.startswith("interval:"):
                # Parse interval
                interval_str = task.schedule.replace("interval:", "")
                interval_parts = dict(part.split("=") for part in interval_str.split(","))
                interval_kwargs = {k: int(v) for k, v in interval_parts.items()}
                
                self.scheduler.add_job(
                    lambda t=task: self._execute_task(t),
                    IntervalTrigger(**interval_kwargs),
                    id=task.task_id,
                    replace_existing=True
                )
            else:
                # Cron expression
                self.scheduler.add_job(
                    lambda t=task: self._execute_task(t),
                    CronTrigger.from_crontab(task.schedule),
                    id=task.task_id,
                    replace_existing=True
                )
            
            logger.info(f"Added task {task.name} to scheduler: {task.schedule}")
            
        except Exception as e:
            logger.error(f"Error adding task to scheduler: {str(e)}")
    
    def start(self):
        """Bắt đầu scheduler"""
        if self.scheduler:
            # Add all registered tasks
            for task in self.registry.get_all_tasks():
                if task.enabled:
                    self._add_to_scheduler(task)
            
            self.scheduler.start()
            self.is_running = True
            logger.info("Scheduler started")
        else:
            logger.warning("No scheduler available")
    
    def stop(self):
        """Dừng scheduler"""
        if self.scheduler:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("Scheduler stopped")
    
    def run_task(self, task_id: str) -> Optional[TaskResult]:
        """Chạy task ngay lập tức (manual trigger)"""
        task = self.registry.get_task(task_id)
        if not task:
            logger.error(f"Task not found: {task_id}")
            return None
        
        return self._execute_task(task)
    
    def run_all_tasks(self) -> List[TaskResult]:
        """Chạy tất cả enabled tasks"""
        results = []
        for task in self.registry.get_all_tasks():
            if task.enabled:
                result = self._execute_task(task)
                results.append(result)
        return results
    
    def get_status(self) -> Dict:
        """Lấy status của scheduler"""
        tasks = self.registry.get_all_tasks()
        return {
            'is_running': self.is_running,
            'scheduler_type': 'apscheduler' if self.scheduler else 'manual',
            'total_tasks': len(tasks),
            'enabled_tasks': sum(1 for t in tasks if t.enabled),
            'tasks': [
                {
                    'task_id': t.task_id,
                    'name': t.name,
                    'schedule': t.schedule,
                    'enabled': t.enabled,
                    'run_count': t.run_count,
                    'success_count': t.success_count,
                    'failure_count': t.failure_count,
                    'last_run': t.last_run.isoformat() if t.last_run else None
                }
                for t in tasks
            ]
        }


# ============= PRE-DEFINED TASKS =============

def create_data_fetch_task(api_base_url: str, symbols: List[str] = None) -> Callable:
    """
    Tạo task fetch data từ API
    
    Args:
        api_base_url: Base URL của API (e.g., "https://kltn-stock-api.onrender.com")
        symbols: Danh sách mã cổ phiếu (None = fetch all VN30)
    """
    def fetch_data():
        logger.info("Starting data fetch task...")
        
        try:
            # Fetch data
            response = requests.post(
                f"{api_base_url}/api/data/fetch-all",
                params={'days': 7},
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Data fetch completed: {result}")
                return result
            else:
                raise Exception(f"API returned {response.status_code}")
        
        except Exception as e:
            logger.error(f"Data fetch failed: {str(e)}")
            raise
    
    return fetch_data


def create_health_check_task(api_base_url: str) -> Callable:
    """Tạo task health check"""
    def health_check():
        try:
            response = requests.get(f"{api_base_url}/api/health", timeout=30)
            return {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'response_time_ms': response.elapsed.total_seconds() * 1000
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    return health_check


def create_model_training_task(api_base_url: str, symbol: str, model_type: str = 'ensemble') -> Callable:
    """Tạo task training model"""
    def train_model():
        logger.info(f"Starting model training for {symbol}...")
        
        try:
            response = requests.post(
                f"{api_base_url}/api/predictions/predict",
                json={
                    'symbol': symbol,
                    'model_type': model_type,
                    'days_ahead': 5
                },
                timeout=600  # 10 minutes
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Training failed: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    return train_model


def create_notification_task(webhook_url: str, message_template: str) -> Callable:
    """Tạo task gửi notification (Slack, Discord, etc.)"""
    def send_notification():
        message = message_template.format(
            timestamp=datetime.now().isoformat(),
            date=datetime.now().strftime('%Y-%m-%d')
        )
        
        try:
            response = requests.post(
                webhook_url,
                json={'message': message, 'timestamp': datetime.now().isoformat()},
                timeout=30
            )
            return {'status': 'sent', 'response_code': response.status_code}
        
        except Exception as e:
            logger.error(f"Notification failed: {str(e)}")
            raise
    
    return send_notification


# ============= RENDER CRON JOBS =============

def generate_render_cron_config(api_base_url: str) -> str:
    """
    Generate cấu hình cron jobs cho Render.com
    
    Thêm vào render.yaml:
    """
    config = f'''
# Add this to your render.yaml

services:
  # Main API service
  - type: web
    name: kltn-stock-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn src.api_v2:app --host 0.0.0.0 --port $PORT

# Cron Jobs
cron:
  # Daily data fetch - 7:00 AM UTC (2:00 PM Vietnam)
  - name: daily-data-fetch
    schedule: "0 7 * * 1-5"
    command: |
      curl -X POST "{api_base_url}/api/data/fetch-all?days=7"
    
  # Weekly full update - Sunday 00:00 UTC
  - name: weekly-full-update
    schedule: "0 0 * * 0"
    command: |
      curl -X POST "{api_base_url}/api/data/fetch-all?days=30"
    
  # Health check - Every 10 minutes
  - name: health-check
    schedule: "*/10 * * * *"
    command: |
      curl -s "{api_base_url}/api/health" || echo "Health check failed"
'''
    return config


# ============= FASTAPI ENDPOINTS FOR SCHEDULER =============

def create_scheduler_router():
    """
    Tạo FastAPI router cho scheduler endpoints
    Import và include trong main API file
    """
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel
    
    router = APIRouter(prefix="/api/scheduler", tags=["Scheduler"])
    
    # Global scheduler instance
    scheduler_service = SchedulerService(use_apscheduler=APSCHEDULER_AVAILABLE)
    
    class TaskCreate(BaseModel):
        name: str
        schedule: str  # Cron expression
        description: str = ""
        enabled: bool = True
    
    @router.get("/status")
    def get_scheduler_status():
        """Lấy status của scheduler"""
        return scheduler_service.get_status()
    
    @router.post("/tasks/{task_id}/run")
    def run_task(task_id: str):
        """Chạy task ngay lập tức"""
        result = scheduler_service.run_task(task_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return result.to_dict()
    
    @router.get("/results")
    def get_task_results(task_id: str = None, limit: int = 100):
        """Lấy kết quả các task đã chạy"""
        results = scheduler_service.registry.get_results(task_id, limit)
        return [r.to_dict() for r in results]
    
    @router.post("/start")
    def start_scheduler():
        """Start scheduler"""
        scheduler_service.start()
        return {"message": "Scheduler started"}
    
    @router.post("/stop")
    def stop_scheduler():
        """Stop scheduler"""
        scheduler_service.stop()
        return {"message": "Scheduler stopped"}
    
    return router


# Quick setup function
def setup_default_scheduler(api_base_url: str) -> SchedulerService:
    """
    Setup scheduler với các tasks mặc định
    
    Args:
        api_base_url: Base URL của API
    
    Returns:
        Configured SchedulerService
    """
    service = SchedulerService()
    
    # Task 1: Daily data fetch (7AM UTC = 2PM Vietnam)
    service.add_task(
        name="Daily Data Fetch",
        function=create_data_fetch_task(api_base_url),
        schedule="0 7 * * 1-5",  # Monday to Friday
        description="Fetch stock data daily at 7AM UTC",
        priority=TaskPriority.HIGH
    )
    
    # Task 2: Health check (every 10 minutes)
    service.add_task(
        name="Health Check",
        function=create_health_check_task(api_base_url),
        schedule="interval:minutes=10",
        description="Check API health every 10 minutes",
        priority=TaskPriority.LOW
    )
    
    logger.info("Default scheduler setup completed")
    return service


if __name__ == "__main__":
    print("Scheduler Service for Stock Prediction API")
    print("=" * 60)
    print("\nSupported scheduling methods:")
    print("1. APScheduler (Python-based, runs within API)")
    print("2. Render Cron Jobs (render.yaml configuration)")
    print("3. UptimeRobot (Keep-alive, miễn phí)")
    print("\nUsage:")
    print("  service = setup_default_scheduler('https://your-api.com')")
    print("  service.start()")
    print("\nCron Expression Examples:")
    print("  '0 7 * * 1-5'  - 7:00 AM, Mon-Fri")
    print("  '0 0 * * 0'    - Midnight, Sunday")
    print("  '*/10 * * * *' - Every 10 minutes")
