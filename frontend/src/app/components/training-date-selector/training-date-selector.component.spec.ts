import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TrainingDateSelectorComponent } from './training-date-selector.component';

describe('TrainingDateSelectorComponent', () => {
  let component: TrainingDateSelectorComponent;
  let fixture: ComponentFixture<TrainingDateSelectorComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [TrainingDateSelectorComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(TrainingDateSelectorComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
