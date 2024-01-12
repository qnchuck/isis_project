export class ForecastData{
    datetime:string;
    Load:number;
    constructor(datetime:string, load:number){
        this.datetime = datetime,
        this.Load = load
    }
}