import os 

UPLOAD_FOLDER = '/home/qnchuck/Desktop/isis/backend/uploads/'
ALLOWED_EXTENSIONS = {'csv'}

class FilesService:           
        
    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def determine_subfolder(self,filename):
        if 'New York City' in filename:
            return 'weather'
        elif 'pal.csv' in filename:
            return 'load'
        else:
            return 'other'


    # Function to check if the file extension is allowed
    def allowed_file(self, filename):
        # You can add more allowed extensions if needed
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}
    
    def save_file_to_specific_directory(self, file):
         # Check if the file is present and has an allowed extension
        if file and self.allowed_file(file.filename):
        # Create subfolder based on filename
            subfolder = self.determine_subfolder(file.filename)

            # Create subfolder if it doesn't exist
        
            # Save the file in the determined subfolder
            if file and self.allowed_file(file.filename):
                filename = os.path.join(UPLOAD_FOLDER+subfolder, file.filename)
                file.save(filename)
                print(f'File uploaded successfully: {filename}')
                return True
            
        return False

