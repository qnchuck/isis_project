import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
@Injectable({
  providedIn: 'root'
})
export class ForecastService {
  private baseUrl = 'http://localhost:5000/prediction'; // Replace with your backend API URL

  constructor(private http: HttpClient) {}

  startForecast(date: string, days: number): Observable<any> {
    // Adjust the API endpoint and parameters based on your backend implementation
    const url = `${this.baseUrl}/predict?date=${date}&days=${days}`;
    return this.http.get(url);
  }
}
