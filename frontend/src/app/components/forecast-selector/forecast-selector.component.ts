import { Component, Output, EventEmitter } from '@angular/core';
import { ForecastService } from '../../services/forecast.service';
import { MatTableDataSource } from '@angular/material/table';
import { DatePipe } from '@angular/common';
import { ForecastData } from '../../models/forecast';
@Component({
  selector: 'app-forecast-selector',
  templateUrl: './forecast-selector.component.html',
  styleUrl: './forecast-selector.component.css'
})
export class ForecastSelectorComponent {
  selectedDate: string = '2021-08-10';
  numberOfDays: number = 1;
  forecastData: any;// ForecastData[] = [new ForecastData('2021-01-01',123),new ForecastData('2021-01-01',123)];

  displayedColumns: string[] = ['datetime', 'Load'];

  constructor(private forecastService: ForecastService) { 
    this.forecastData = [];
    // this.startForecast()
  }


  startForecast(): void {
    this.forecastService.startForecast(this.selectedDate, this.numberOfDays).subscribe(
      (data) => {
        const responseObj = JSON.parse(data.res);

        // Assuming 'res' contains the array of forecast data
        this.forecastData = responseObj;
        console.log(this.forecastData);
        console.log(this.forecastData);
      },
      (error) => {
        console.error('Error fetching forecast:', error);
      }
    );
  }
}
 
