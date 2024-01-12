import { Component, OnInit} from '@angular/core';
import { TestService } from './services/test.service';
import { TestData } from './models/data';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  newdata:TestData = new TestData('','');

  constructor(private _apiservice:TestService) {
  }

  ngOnInit() {
	// this.getData();
  }

  getData() {
	this._apiservice.getdata().subscribe(res=>{
  	this.newdata=res;
	})
  }

  handleForecastStart(event: { date: string, days: number }) {
    // Implement your logic to start the forecast using the provided date and number of days.
    console.log('Starting forecast for', event);
  }
}