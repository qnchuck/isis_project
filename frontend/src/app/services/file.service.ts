import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class FileService {
  private apiUrl = 'http://127.0.0.1:5000/file_controller'; 

  constructor(private http: HttpClient) {}

  sendFolderPath(folderPath: string): Observable<any> {
    const data = { folderPath };
    console.log(folderPath)
    return this.http.post(`${this.apiUrl}/folder_path`, data);
  }
}
