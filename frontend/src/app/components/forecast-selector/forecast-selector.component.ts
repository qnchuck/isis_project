import { Component, Output, EventEmitter } from '@angular/core';
import { ForecastService } from '../../services/forecast.service';
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
  filterDate: string = '';
  displayedColumns: string[] = ['datetime', 'Load'];
  modelNames: string[] = [];
  selectedModel: string = ''; // Add a property to store the selected model

  constructor(private forecastService: ForecastService) { 
    this.forecastData = [];
    this.numberOfDays = 1;
    // this.startForecast()
  }
  ngOnInit(): void {
    this.fetchModelNames();
  }
  onSelectDropdownOpen(isOpen: boolean): void {
    if (isOpen) {
      // The dropdown is opened, you can perform your logic here
      this.fetchModelNames();
    }
  }
  
  checkForNewModels(): void {
    // Your logic to check for new models and update the modelNames array
    // For example, you can make an HTTP request to get the latest model names
  }
  
  fetchModelNames(): void {
    this.forecastService.getModelNames().subscribe(
      (names:any) => {
        this.modelNames = names;
      },
      (error:string) => {
        console.error('Error fetching model names:', error);
      }
    );
  }
  onModelSelect(): void {
    console.log('Selected Model:', this.selectedModel);
  }
  startForecast(): void {
    this.forecastService.startForecast(this.selectedDate, this.numberOfDays, this.selectedModel).subscribe(
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
 
