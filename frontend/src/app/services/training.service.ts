import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class TrainingService {

  private apiUrl = 'http://localhost:5000/training'; // Update with your Flask API URL

  constructor(private http: HttpClient) {}

  startTraining(startDate: string, endDate: string): Observable<any> {
    const data = { startDate, endDate };
    return this.http.post(`${this.apiUrl}/train_model`, data);
  }
  doPreprocessing(startDate: string, endDate: string):Observable<any>{
    const data = { startDate, endDate };
    return this.http.post(`${this.apiUrl}/preprocess`, data);
  }
}
