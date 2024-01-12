import { Component } from '@angular/core';
import { FileService } from '../../services/file.service';

@Component({
  selector: 'app-folder-browser',
  templateUrl: './folder-browser.component.html',
  styleUrl: './folder-browser.component.css'
})
export class FolderBrowserComponent {
  folderPath: string = '';

  constructor(private folderService: FileService) {}

  onFolderSelected(event: any): void {
    this.folderPath = event.target.value;
  }

  sendFolderPath(): void {
    if (this.folderPath) {
      this.folderService.sendFolderPath(this.folderPath).subscribe(
        (response) => {
          console.log('Folder path sent successfully:', response);
        },
        (error) => {
          console.error('Error sending folder path:', error);
        }
      );
    }
  }
}
