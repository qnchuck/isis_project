import { Component } from '@angular/core';
import { TrainingService } from '../../services/training.service';
import { MatDialog } from '@angular/material/dialog';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { Inject } from '@angular/core';

@Component({
  selector: 'app-training-date-selector',
  templateUrl: './training-date-selector.component.html',
  styleUrls: ['./training-date-selector.component.css']
})
export class TrainingDateSelectorComponent {
  selectedDateFrom: string | undefined = "2018-01-01";
  selectedDateTo: string | undefined = "2021-09-05";

  constructor(private dateService: TrainingService, private dialog: MatDialog) { }

  openDialog(message:string): void {
    this.dialog.open(DialogContentComponent, {
      data: { 'message': message },
    });
  }

  sendDatesForTraining(): void {
    if (this.isYearValid(this.selectedDateFrom) && this.isYearValid(this.selectedDateTo)) {
      if (this.selectedDateFrom && this.selectedDateTo) {
        if (this.isDateFromBeforeDateTo()) {
          this.dateService
            .startTraining(this.selectedDateFrom, this.selectedDateTo)
            .subscribe(
              (response) => {
                this.openDialog("Training done succesfully");
                console.log('Response from server:', response);
              },
              (error) => {
                console.error('Error:', error);
              }
            );
        } else {
          this.openDialog('Start date must be before end date.');
        }
      } else {
        this.openDialog('Please select both start and end dates.');
      }
    } else {
      this.openDialog('Selected year should not be 2020.');
    }
  }

  preprocessData(): void {
    if (this.isYearValid(this.selectedDateFrom) && this.isYearValid(this.selectedDateTo)) {
      if (this.selectedDateFrom && this.selectedDateTo) {
        if (this.isDateFromBeforeDateTo()) {
          this.dateService
            .doPreprocessing(this.selectedDateFrom, this.selectedDateTo)
            .subscribe(
              (response) => {
                this.openDialog('Training done!');
                console.log('Response from server:', response);
              },
              (error) => {
                console.error('Error:', error);
              }
            );
        } else {
          this.openDialog('Start date must be before end date.');
        }
      } else {
        this.openDialog('Please select both start and end dates.');
      }
    } else {
      this.openDialog('Selected year should not be 2020.');
    }
  }

  private isDateFromBeforeDateTo(): boolean {
    return new Date(this.selectedDateFrom!) < new Date(this.selectedDateTo!);
  }

  private isYearValid(date: string | undefined): boolean {
    return date ? new Date(date).getFullYear() !== 2020 : false;
  }
}

@Component({
  selector: 'app-dialog-content',
  template: ` <div style="border: 2px solid #ccc; padding: 20px; border-radius: 5px; width: 300px; max-width: 100%; box-sizing: border-box;">
  <p>{{ data.message }}</p>
</div>
`,
})
export class DialogContentComponent {
  constructor(@Inject(MAT_DIALOG_DATA) public data: any) { }
}
