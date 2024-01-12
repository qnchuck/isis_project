import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FolderBrowserComponent } from './folder-browser.component';

describe('FolderBrowserComponent', () => {
  let component: FolderBrowserComponent;
  let fixture: ComponentFixture<FolderBrowserComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [FolderBrowserComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(FolderBrowserComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
