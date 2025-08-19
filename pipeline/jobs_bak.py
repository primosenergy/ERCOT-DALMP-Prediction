from datetime import date, timedelta
from typing import Optional, Union

class Jobs:
    """Coordinates Scrapes ➜ Loads steps."""
    def __init__(self, scrapes, loads, weather_points):
        self.scrapes = scrapes
        self.loads = loads
        self.weather_points = weather_points
        self.latest_session_id: str | None = None

    async def run_session(self):
        sid = await self.scrapes.get_session_id()
        self.loads.upsert_session(sid)
        self.latest_session_id = sid
        return sid

    async def run_prices(self):
        if not self.latest_session_id:
            await self.run_session()
        csv_bytes = await self.scrapes.get_prices_tomorrow(self.latest_session_id)
        self.loads.load_prices(csv_bytes)

    async def run_load_fcst(self,
                            from_operating_date: Optional[Union[str, date]] = None,
                            to_operating_date: Optional[Union[str, date]] = None):
        if not self.latest_session_id:
            await self.run_session()
        if from_operating_date is None and to_operating_date is None:
            from_operating_date = date.today()
            to_operating_date = date.today()
        elif to_operating_date is None:
            to_operating_date = from_operating_date

        json_bytes = await self.scrapes.get_load_forecast(self.latest_session_id, from_operating_date, to_operating_date)
        self.loads.load_load_fcst_json(json_bytes)

    async def run_hist_weather(self):
        data = await self.scrapes.get_hist_weather(self.weather_points)
        self.loads.load_hist_weather(data)

    async def run_fcst_weather(self):
        target = date.today() + timedelta(days=1)
        data = await self.scrapes.get_forecast_weather(self.weather_points, target)
        self.loads.load_fcst_weather(data)
