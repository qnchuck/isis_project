import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, catchError, forkJoin } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class FileService {
  private apiUrl = 'http://127.0.0.1:5000/file_controller'; 

  constructor(private http: HttpClient) {}

  upload(file:any): Observable<any> {
    const formData = new FormData();
    formData.append("file", file, file.name);
  
    return this.http.post(this.apiUrl+'/upload', formData).pipe(
      catchError((error) => {
        console.error('Error uploading file:', error);
        throw error;
      })
    );
  }
  uploadMultiple(files: File[]): Observable<any[]> {
    const observables: Observable<any>[] = [];

    files.forEach(file => {
      const formData = new FormData();
      formData.append('file', file, file.name);

      const observable = this.http.post(this.apiUrl+'/upload', formData);
      observables.push(observable);
    });

    return forkJoin(observables);
  }
}
