import { Component, Output, EventEmitter } from '@angular/core';
import { ForecastService } from '../../services/forecast.service';
@Component({
  selector: 'app-forecast-selector',
  templateUrl: './forecast-selector.component.html',
  styleUrl: './forecast-selector.component.css'
})
export class ForecastSelectorComponent {
  selectedDate: string = '';
  numberOfDays: number = 1;
  forecastData: any;

  constructor(private forecastService: ForecastService) { }

  startForecast(): void {
    this.forecastService.startForecast(this.selectedDate, this.numberOfDays).subscribe(
      (data) => {
        this.forecastData = data;
        console.log('Forecast Data:', this.forecastData);
      },
      (error) => {
        console.error('Error fetching forecast:', error);
      }
    );
  }
}
 
