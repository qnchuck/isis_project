// folder-browser.component.ts
import { Component, OnInit } from '@angular/core';
import { FileService } from '../../services/file.service';
import { MatIcon } from '@angular/material/icon';
@Component({
  selector: 'app-folder-browser',
  templateUrl: './folder-browser.component.html',
  styleUrls: ['./folder-browser.component.css']
})
export class FolderBrowserComponent implements OnInit {
  shortLinks: string[] = [];
  loading: boolean = false;
  files: File[] = [];

  constructor(private fileUploadService: FileService) {
    this.files = [];
  }

  ngOnInit(): void {}

  onChange(event: any) {
    this.loading = false;

    // Check if the 'webkitdirectory' attribute is supported
    const supportsWebkitDirectory = 'webkitdirectory' in HTMLInputElement.prototype;

    // Reset the files array on each change
    this.files = [];

    if (supportsWebkitDirectory && event.target?.files?.length) {
      // If 'webkitdirectory' is supported, use it to get the files in the directory
      const directoryInput = event.target;
      const filesInDirectory = Array.from(directoryInput.files);
      this.files = filesInDirectory as File[];
    } else if (event.target?.files) {
      // Fallback for browsers that don't support 'webkitdirectory'
      for (let i = 0; i < event.target.files.length; i++) {
        this.files.push(event.target.files[i]);
      }
    }
  }

  onUpload() {
    this.loading = true;
    console.log(this.files);

    this.fileUploadService.uploadMultiple(this.files).subscribe(
      (events: any[]) => {
        this.shortLinks = events.map(event => event.link);
        this.loading = false;
      },
      (error) => {
        console.error('Error during upload:', error);
        this.loading = false;
      }
    );
  }
}
