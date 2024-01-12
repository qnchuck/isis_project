import { Component } from '@angular/core';
import { TrainingService } from '../../services/training.service';

@Component({
  selector: 'app-training-date-selector',
  templateUrl: './training-date-selector.component.html',
  styleUrl: './training-date-selector.component.css'
})
export class TrainingDateSelectorComponent {
  selectedDateFrom: string | undefined;
  selectedDateTo: string | undefined;

  constructor(private dateService: TrainingService) {}

  sendDatesForTraining(): void {
    if (this.selectedDateFrom && this.selectedDateTo) {
      if (this.isDateFromBeforeDateTo()) {
        this.dateService
          .startTraining(this.selectedDateFrom, this.selectedDateTo)
          .subscribe(
            (response) => {
              console.log('Response from server:', response);
              // Perform further actions with the response
            },
            (error) => {
              console.error('Error:', error);
            }
          );
      } else {
        console.warn('Start date must be before end date.');
      }
    } else {
      console.warn('Please select both start and end dates.');
    }
  }

  private isDateFromBeforeDateTo(): boolean {
    return new Date(this.selectedDateFrom!) < new Date(this.selectedDateTo!);
  }
}