from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import logging
import uuid
import threading
import json
import os
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from core.registry import get_registry, ComponentRegistry
from core.types import ConversationResult

logger = logging.getLogger(__name__)


class BaseSession(ABC):
    def __init__(self, component_type: str, name: str, config: Optional[Dict[str, Any]] = None):
        self.component_type = component_type
        self.name = name
        self.config = config or {}

        self.start_time = datetime.now()
        self.num_conversationts = self.config.get("num_conversations", 1)
        self.registry = get_registry()
        
        self._lock = threading.RLock()
        self._task_queue = Queue()
        self._processing_set = set()
        self._completed_set = set()
        self._failed_set = set()
        
        self.save_batch_size = self.config.get("save_batch_size")
        self.max_workers = self.config.get("max_workers")
        self.output_file = self._setup_output_file()
        self._results_buffer: List[Dict[str, Any]] = []
        self._results_lock = threading.Lock()
        self._pbar = None
        self._pbar_lock = threading.Lock()
        
        self._write_queue = Queue()
        self._writer_thread = None
        self._stop_writer = threading.Event()

        self.initialize(self.num_conversationts, self.registry)
        
        for i in range(self.num_conversationts):
            self._task_queue.put(i)
        logger.debug(f"Session {self.name} initialized, managing {self.num_conversationts} conversations, thread pool size: {self.max_workers}")


    @abstractmethod
    def initialize(self, num_conversation: int, registry: ComponentRegistry):
        pass

    @abstractmethod
    def start_conversation(self, index: int) -> ConversationResult:
        pass

    @abstractmethod
    def custom_result(self, conversation_result: ConversationResult, index: int) -> Dict[str, Any]:
        pass
    
    def post_process(self, result: Dict[str, Any]) -> None:
        """
        Post-processing method, called after all conversations are executed.
        Subclasses can override this method to implement custom post-processing logic (e.g., generating reports).

        Args:
            result: Result dictionary returned by run_all_conversations
        """
        pass


    def _setup_output_file(self) -> str:
        output_dir = self.config.get("output_dir", "outputs")
        if not os.path.isabs(output_dir):
            output_dir = os.path.abspath(output_dir)
        
        now = datetime.now()
        date_dir = now.strftime("%Y-%m-%d")
        
        full_output_dir = os.path.join(output_dir, date_dir)
        if not os.path.exists(full_output_dir):
            os.makedirs(full_output_dir, exist_ok=True)
        
        return os.path.join(full_output_dir, self._generate_config_identifier())
    
    def _generate_config_identifier(self) -> str:
        num_conversations = self.config.get("num_conversations", 1)
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"conversation_{num_conversations}_sample_{timestamp}.jsonl"

    def _writer_worker(self):
        """Dedicated writer thread for async file I/O"""
        while not self._stop_writer.is_set() or not self._write_queue.empty():
            try:
                results = self._write_queue.get(timeout=0.1)
                if results:
                    self._save_results_to_file(results)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Writer thread exception: {e}", exc_info=True)

    def _save_results_to_file(self, results: List[Dict[str, Any]]) -> None:
        if not results:
            return
        
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                for result in results:
                    json_line = json.dumps(result, ensure_ascii=False)
                    f.write(json_line + '\n')
            logger.info(f"Saved {len(results)} results to {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}", exc_info=True)
    
    def _flush_buffer_if_needed(self, force: bool = False) -> None:
        """Reduce lock hold time, move file I/O outside the lock"""
        results_to_save = None
        
        with self._results_lock:
            should_flush = force or len(self._results_buffer) >= self.save_batch_size
            if should_flush and self._results_buffer:
                results_to_save = self._results_buffer.copy()
                self._results_buffer.clear()
        
        # Perform file I/O outside the lock (via async queue)
        if results_to_save:
            self._write_queue.put(results_to_save)
            if force:
                logger.debug(f"Flushing remaining {len(results_to_save)} results")
    
    def _add_result_to_buffer(self, conversation_result: ConversationResult, index: int) -> None:
        """Reduce lock acquisitions, perform time-consuming operations outside the lock"""
        # Perform time-consuming formatting outside the lock
        formatted_result = self.custom_result(conversation_result, index=index)
        formatted_result['conversation_index'] = index
        
        should_flush = False
        with self._results_lock:
            self._results_buffer.append(formatted_result)
            should_flush = len(self._results_buffer) >= self.save_batch_size
        
        # Check outside the lock whether flush is needed
        if should_flush:
            self._flush_buffer_if_needed()
    
    def _mark_conversation_status(self, index: int, status: str) -> None:
        """Reduce lock nesting, quickly get data inside the lock then update progress bar outside"""
        completed = failed = processing = 0
        
        with self._lock:
            if status == 'processing':
                self._processing_set.add(index)
            elif status == 'completed':
                self._processing_set.discard(index)
                self._completed_set.add(index)
            elif status == 'failed':
                self._processing_set.discard(index)
                self._failed_set.add(index)
            
            # Quickly get data inside the lock
            if status in ('completed', 'failed'):
                completed = len(self._completed_set)
                failed = len(self._failed_set)
                processing = len(self._processing_set)
        
        # Update progress bar outside the lock to avoid lock nesting
        if status in ('completed', 'failed'):
            self._update_progress_bar_with_data(completed, failed, processing)
    
    def _update_progress_bar_with_data(self, completed: int, failed: int, processing: int) -> None:
        """Update progress bar with pre-computed data to avoid lock nesting"""
        if self._pbar is not None:
            with self._pbar_lock:
                self._pbar.n = completed + failed
                self._pbar.set_postfix({
                    '✅': completed,
                    '❌': failed,
                    '🔄': processing
                })
                self._pbar.refresh()
    
    def _update_progress_bar(self) -> None:
        """Update progress bar (kept for compatibility)"""
        if self._pbar is not None:
            with self._lock:
                completed = len(self._completed_set)
                failed = len(self._failed_set)
                processing = len(self._processing_set)
            
            self._update_progress_bar_with_data(completed, failed, processing)
    
    def _worker(self):
        while True:
            try:
                index = self._task_queue.get_nowait()
            except Empty:
                break
            
            self._mark_conversation_status(index, 'processing')
            
            try:
                result = self.start_conversation(index=index)
                
                self._add_result_to_buffer(result, index)
                self._mark_conversation_status(index, 'completed')
                    
            except Exception as e:
                logger.error(f"Conversation {index} execution failed: {e}", exc_info=True)
                self._mark_conversation_status(index, 'failed')
    
    def run_all_conversations(self, progress_callback: Optional[Callable[[int, int, int], None]] = None) -> Dict[str, Any]:
        logger.debug(f"Starting {self.num_conversationts} conversations with {self.max_workers} threads")
        start_time = datetime.now()

        # Start async writer thread
        self._stop_writer.clear()
        self._writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self._writer_thread.start()
        
        # Initialize tqdm progress bar
        self._pbar = tqdm(
            total=self.num_conversationts,
            desc="Conversation Progress",
            unit="conv",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._worker) for _ in range(self.max_workers)]
                
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Worker thread exception: {e}", exc_info=True)
        finally:
            # Close progress bar
            if self._pbar is not None:
                self._pbar.close()
                self._pbar = None
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Flush remaining buffer
        self._flush_buffer_if_needed(force=True)
        
        # Wait for writer thread to complete all file I/O
        self._stop_writer.set()
        if self._writer_thread:
            self._writer_thread.join(timeout=30)
            if self._writer_thread.is_alive():
                logger.warning("Writer thread did not finish within 30 seconds")
        
        result = {
            "total": self.num_conversationts,
            "completed": len(self._completed_set),
            "failed": len(self._failed_set),
            "duration_seconds": duration,
            "completed_indices": self._get_sorted_indices(self._completed_set),
            "failed_indices": self._get_sorted_indices(self._failed_set),
            "output_file": self.output_file
        }
        
        logger.debug(f"All conversations completed! Success: {result['completed']}, Failed: {result['failed']}, Duration: {duration:.2f}s")
        logger.debug(f"Results saved to: {self.output_file}")

        # Call post-processing method (subclasses can override)
        self.post_process(result)
        
        return result
    
    def _get_sorted_indices(self, index_set: set) -> List[int]:
        return sorted(list(index_set))
    
    
    
    def get_progress_info(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total": self.num_conversationts,
                "pending": self._task_queue.qsize(),
                "processing": len(self._processing_set),
                "completed": len(self._completed_set),
                "failed": len(self._failed_set),
                "processing_indices": self._get_sorted_indices(self._processing_set),
                "completed_indices": self._get_sorted_indices(self._completed_set),
                "failed_indices": self._get_sorted_indices(self._failed_set)
            }
    
