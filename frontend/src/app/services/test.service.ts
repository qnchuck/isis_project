import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { TestData } from '../models/data';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class TestService {
  retval: Observable<TestData>;
    constructor(private _http: HttpClient) { 
      this.retval = new Observable<TestData>;
    }
  
  getdata():Observable<TestData>{
    return this._http.get<TestData>('http://127.0.0.1:5000/api/data');
     
  }
  
}
