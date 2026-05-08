import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

// API 基础地址
// Docker / 生产环境下建议通过反向代理访问同域 `/api`
const API_BASE_URL = '/api';

export interface CollectionRequest {
  mode: string;  // 'sample' | 'custom' | 'chinese_reviews' | 'steamspy' | 'top_games'
  delay?: number;
  skipSteamcharts?: boolean;
  appIds?: number[];
  threshold?: number;
  maxGames?: number;
  limit?: number;
}

export interface TaskResponse {
  task_id: string;
  message: string;
}

export interface TaskStatus {
  id: string;
  type: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  message: string;
  logs: Array<{
    timestamp: string;
    message: string;
    level: 'info' | 'success' | 'error' | 'warning';
  }>;
  result?: any;
  error?: string;
  created_at: string;
  updated_at: string;
}

export interface ReviewRequest {
  gameName?: string;
  appId: number;
  maxReviews?: number;
  language?: string;
  reviewType?: string;
}

export interface TrainingRequest {
  inputFile: string;
  testSize?: number;
  randomState?: number;
  nEstimators?: number;
}

export interface CleaningRequest {
  inputFile: string;
  useApi?: boolean;
  useMl?: boolean;
  useEstimation?: boolean;
  deleteFailed?: boolean;
}

@Injectable({
  providedIn: 'root'
})
export class SteamApiService {
  constructor(private http: HttpClient) { }

  // ==================== 游戏数据采集 ====================

  /**
   * 开始游戏数据采集任务
   */
  startCollection(request: CollectionRequest): Observable<TaskResponse> {
    return this.http.post<TaskResponse>(`${API_BASE_URL}/collect/start`, request);
  }

  /**
   * 获取采集任务状态
   */
  getCollectionStatus(taskId: string): Observable<TaskStatus> {
    return this.http.get<TaskStatus>(`${API_BASE_URL}/collect/status/${taskId}`);
  }

  /**
   * 取消采集任务
   */
  cancelCollection(taskId: string): Observable<any> {
    return this.http.delete(`${API_BASE_URL}/tasks/${taskId}`);
  }

  /**
   * 下载采集结果
   */
  downloadCollectionResult(taskId: string): string {
    return `${API_BASE_URL}/collect/download/${taskId}`;
  }

  downloadCollectionJSON(taskId: string): string {
    return `${API_BASE_URL}/collect/download/${taskId}?format=json`;
  }

  getCollectionResult(taskId: string): Observable<Blob> {
    return this.http.get(`${API_BASE_URL}/collect/download/${taskId}`, {
      responseType: 'blob'
    });
  }

  // ==================== 评论采集 ====================

  /**
   * 开始评论采集任务
   */
  startReviewCollection(request: ReviewRequest): Observable<TaskResponse> {
    return this.http.post<TaskResponse>(`${API_BASE_URL}/reviews/start`, request);
  }

  /**
   * 获取评论采集任务状态
   */
  getReviewCollectionStatus(taskId: string): Observable<TaskStatus> {
    return this.http.get<TaskStatus>(`${API_BASE_URL}/reviews/status/${taskId}`);
  }

  /**
   * 取消评论采集任务
   */
  cancelReviewCollection(taskId: string): Observable<any> {
    return this.http.delete(`${API_BASE_URL}/tasks/${taskId}`);
  }

  /**
   * 下载评论数据
   */
  downloadReviews(taskId: string): string {
    return `${API_BASE_URL}/reviews/download/${taskId}`;
  }

  getReviewsResult(taskId: string): Observable<Blob> {
    return this.http.get(`${API_BASE_URL}/reviews/download/${taskId}`, {
      responseType: 'blob'
    });
  }

  // ==================== 模型训练 ====================

  /**
   * 开始模型训练任务
   */
  startTraining(request: TrainingRequest): Observable<TaskResponse> {
    return this.http.post<TaskResponse>(`${API_BASE_URL}/train/start`, request);
  }

  /**
   * 获取训练任务状态
   */
  getTrainingStatus(taskId: string): Observable<TaskStatus> {
    return this.http.get<TaskStatus>(`${API_BASE_URL}/train/status/${taskId}`);
  }

  /**
   * 取消训练任务
   */
  cancelTraining(taskId: string): Observable<any> {
    return this.http.delete(`${API_BASE_URL}/tasks/${taskId}`);
  }

  /**
   * 获取训练统计信息
   */
  getTrainingStats(taskId: string): Observable<any> {
    return this.http.get(`${API_BASE_URL}/train/stats/${taskId}`);
  }

  /**
   * 下载训练好的模型
   */
  downloadModel(taskId: string, modelName: string): string {
    return `${API_BASE_URL}/train/download/${taskId}/${modelName}`;
  }

  // ==================== 数据清洗 ====================

  /**
   * 开始数据清洗任务
   */
  startCleaning(request: CleaningRequest): Observable<TaskResponse> {
    return this.http.post<TaskResponse>(`${API_BASE_URL}/clean/start`, request);
  }

  /**
   * 获取清洗任务状态
   */
  getCleaningStatus(taskId: string): Observable<TaskStatus> {
    return this.http.get<TaskStatus>(`${API_BASE_URL}/clean/status/${taskId}`);
  }

  /**
   * 取消清洗任务
   */
  cancelCleaning(taskId: string): Observable<any> {
    return this.http.delete(`${API_BASE_URL}/tasks/${taskId}`);
  }

  /**
   * 分析缺失值
   */
  analyzeMissingValues(inputFile: string): Observable<any> {
    const params = new HttpParams().set('inputFile', inputFile);
    return this.http.get(`${API_BASE_URL}/clean/analyze/temp`, { params });
  }

  /**
   * 下载清洗后的数据
   */
  downloadCleanedData(taskId: string): string {
    return `${API_BASE_URL}/clean/download/${taskId}`;
  }

  // ==================== 通用 ====================

  /**
   * 健康检查
   */
  healthCheck(): Observable<any> {
    return this.http.get(`${API_BASE_URL}/health`);
  }

  /**
   * 获取任务状态
   */
  getTaskStatus(taskId: string): Observable<TaskStatus> {
    return this.http.get<TaskStatus>(`${API_BASE_URL}/tasks/${taskId}`);
  }

  /**
   * 取消任务
   */
  cancelTask(taskId: string): Observable<any> {
    return this.http.delete(`${API_BASE_URL}/tasks/${taskId}`);
  }
}
