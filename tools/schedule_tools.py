import re

from config import config
from config.config import PREVIOUS_WEEKDAY, WEEKDAY_MAPPING
from data_pipeline.common_cleaning import parse_spot_length


def _day(value):
    normalized = str(value).strip().lower()
    return WEEKDAY_MAPPING.get(normalized, str(value).strip().title())


def _parse_clock(value):
    match = re.fullmatch(r"(\d{1,2}):([0-5]\d)", str(value).strip())
    if not match:
        return None
    hour, minute = map(int, match.groups())
    if hour > 30:
        return None
    return hour * 60 + minute, hour, minute


def _clock(value, system=None):
    parsed = _parse_clock(value)
    if parsed is None:
        return None
    minute, hour, _ = parsed
    if str(system).strip().lower() != "txline" and hour < 6:
        minute += config.MINUTES_PER_DAY
    return minute


def _channel(value):
    text = str(value).strip()
    return config.CHANNEL_MAPPING.get(text.lower(), text)


def _spot_broadcast_time(spot):
    day = _day(spot.get(config.TRANSMISSION_DAY_COLUMN, ""))
    system = str(spot.get(config.SCHEDULING_SYSTEM_COLUMN, "")).strip()
    parsed = _parse_clock(spot.get(config.BREAK_TIME_COLUMN, ""))
    if parsed is None:
        raise ValueError("Invalid spot break time")
    minute, hour, _ = parsed
    if system.lower() not in {name.lower() for name in config.BROADCAST_CLOCK_SYSTEMS} and hour < 6:
        minute += config.MINUTES_PER_DAY
        day = PREVIOUS_WEEKDAY.get(day, day)
    return day, minute


def _schedule_time(programme):
    system = programme.get(config.SCHEDULING_SYSTEM_COLUMN, "")
    day = _day(programme.get(config.TRANSMISSION_DAY_COLUMN, ""))
    start = _parse_clock(programme.get(config.START_TIME_COLUMN, ""))
    end = _parse_clock(programme.get(config.END_TIME_COLUMN, ""))
    if start is None or end is None:
        return None
    start_minute, start_hour, _ = start
    end_minute, end_hour, _ = end
    if str(system).strip().lower() not in {
            name.lower() for name in config.BROADCAST_CLOCK_SYSTEMS}:
        if start_hour < 6:
            start_minute += config.MINUTES_PER_DAY
            day = PREVIOUS_WEEKDAY.get(day, day)
        if end_hour < 6:
            end_minute += config.MINUTES_PER_DAY
    if end_minute <= start_minute:
        return None
    return day, start_minute, end_minute


def place_spot(spot, schedules):
    day, minute = _spot_broadcast_time(spot)
    channel = _channel(spot.get(config.CHANNEL_COLUMN, ""))
    spot_length = parse_spot_length(spot.get(config.SPOT_LENGTH_COLUMN))
    if spot_length not in config.STANDARD_SPOT_LENGTHS:
        raise ValueError("Unsupported spot length")

    programmes = []
    for programme in schedules:
        if _channel(programme.get(config.CHANNEL_COLUMN, "")) != channel:
            continue
        normalized = _schedule_time(programme)
        if normalized is None:
            continue
        programme_day, start, end = normalized
        if programme_day == day:
            programmes.append((start, end, programme))

    inside = [(start, end, programme) for start, end, programme in programmes
              if start < minute < end]
    if inside:
        selected = sorted(inside, key=lambda row: (row[0], row[1]))[0]
        position = config.BREAK_IN
        neighbouring = [selected[2]]
    else:
        previous = [(start, end, programme) for start, end, programme in programmes
                    if end == minute]
        following = [(start, end, programme) for start, end, programme in programmes
                     if start == minute]
        if not previous or not following:
            raise ValueError("No programme at or adjacent to spot time")
        previous.sort(key=lambda row: row[0])
        following.sort(key=lambda row: row[1])
        position = config.BREAK_ADJACENT
        neighbouring = [previous[-1][2], following[0][2]]

    return {
        "spot_reference": spot.get(config.SPOT_REFERENCE_COLUMN),
        "campaign_reference": spot.get(config.CAMPAIGN_REFERENCE_COLUMN),
        "clock_number": spot.get(config.CLOCK_NUMBER_COLUMN),
        "channel": channel,
        "scheduling_system": spot.get(config.SCHEDULING_SYSTEM_COLUMN),
        "broadcast_day": day,
        "break_time": (f"{minute // config.MINUTES_PER_HOUR:02d}:"
                       f"{minute % config.MINUTES_PER_HOUR:02d}"),
        "break_minute": minute,
        "break_position": position,
        "spot_length_seconds": spot_length,
        "programmes": [dict(programme) for programme in neighbouring],
    }


def _ordinary_minute(value):
    parsed = _parse_clock(value)
    if parsed is None:
        return None
    return parsed[0] % config.MINUTES_PER_DAY


def _flag(programme, column):
    return str(programme.get(column, "")).strip().lower() in {"yes", "true", "1"}


def check_campaign_spots(placements, code):
    results = []
    previous_spots = {}
    ordered = sorted(
        placements,
        key=lambda placement: (
            placement["channel"], placement["broadcast_day"], placement["break_minute"],
            placement["spot_reference"],
        ),
    )
    for placement in ordered:
        breaches = []
        minute = placement["break_minute"]
        programmes = placement["programmes"]
        window = code.get("restricted_hours")
        ordinary_minute = _ordinary_minute(placement.get("break_time"))
        if window:
            start = _ordinary_minute(window["start"])
            end = _ordinary_minute(window["end"])
            if start is None or end is None or ordinary_minute is None:
                raise ValueError("Invalid restricted-hours clock time")
            restricted = (start <= ordinary_minute < end if start < end
                          else ordinary_minute >= start or ordinary_minute < end)
            sport = code.get("live_sport_exception")
            exception = False
            if sport and placement["break_position"] == config.BREAK_IN:
                after = int(sport.get("minutes_after_start", sport.get("after", 0)))
                before = int(sport.get("minutes_before_end", sport.get("before", 0)))
                for programme in programmes:
                    if not _flag(programme, config.LIVE_SPORT_FLAG_COLUMN):
                        continue
                    start_minute = _clock(
                        programme.get(config.START_TIME_COLUMN),
                        programme.get(config.SCHEDULING_SYSTEM_COLUMN),
                    )
                    end_minute = _clock(
                        programme.get(config.END_TIME_COLUMN),
                        programme.get(config.SCHEDULING_SYSTEM_COLUMN),
                    )
                    if (start_minute is not None and end_minute is not None
                            and minute - start_minute > after
                            and end_minute - minute > before):
                        exception = True
                        break
            if restricted and not exception:
                breaches.append(config.BREACH_WINDOW)

        threshold = code.get("index_threshold", code.get("under18_index_threshold"))
        if threshold is not None and any(
                float(programme.get(config.FORECAST_INDEX_COLUMN) or 0) >= float(threshold)
                for programme in programmes):
            breaches.append(config.BREACH_INDEX)

        children_rule = code.get("children_rule")
        children_reached = (children_rule == config.CHILDREN_IN_OR_ADJACENT
                            or (children_rule == config.CHILDREN_IN_ONLY
                                and placement["break_position"] == config.BREAK_IN))
        if children_reached and any(
                _flag(programme, config.MADE_FOR_CHILDREN_FLAG_COLUMN)
                for programme in programmes):
            breaches.append(config.BREACH_CHILDREN)

        key = (placement["campaign_reference"], placement["channel"],
               placement["broadcast_day"])
        prior = previous_spots.setdefault(key, [])
        cap = code.get("daily_cap")
        if cap is not None and len(prior) >= int(cap):
            breaches.append(config.BREACH_FREQUENCY)
        separation = code.get("separation_minutes") or code.get("minimum_separation_minutes")
        if separation and prior and minute - prior[-1] < int(separation):
            breaches.append(config.BREACH_SEPARATION)
        prior.append(minute)

        results.append({
            "spot_reference": placement["spot_reference"],
            "campaign_reference": placement["campaign_reference"],
            "channel": placement["channel"],
            "broadcast_day": placement["broadcast_day"],
            "break_time": placement["break_time"],
            "break_minute": minute,
            "break_position": placement["break_position"],
            "programmes": programmes,
            "breach_codes": breaches,
            "compliant": not breaches,
        })
    return results


def summarise_spot_checks(results):
    breaches = [result for result in results if result["breach_codes"]]
    compliant = [result for result in results if not result["breach_codes"]]
    by_rule = {rule: sum(rule in result["breach_codes"] for result in results)
               for rule in config.BREACH_CODES}
    if not breaches:
        decisions = [config.DECISION_CLEAR, config.DECISION_REJECT]
    elif compliant:
        decisions = [config.DECISION_CLEAR_WITH_RESTRICTIONS, config.DECISION_REJECT]
    else:
        decisions = [config.DECISION_REJECT]
    weekday_order = {day: index for index, day in enumerate(config.WEEKDAYS)}
    first = min(results, key=lambda result: (
        weekday_order.get(result.get("broadcast_day"), len(weekday_order)),
        result.get("break_minute", 0),
    )) if results else None
    return {
        "spots_booked": len(results),
        "spots_compliant": len(compliant),
        "spots_breaching": len(breaches),
        "breaches_by_rule": by_rule,
        "decisions_open": decisions,
        "first_transmission": {
            "broadcast_day": first["broadcast_day"], "break_time": first["break_time"],
        } if first else {},
    }


def compute_clearance_outcome(decision, results):
    if decision not in config.DECISIONS:
        raise ValueError("Unknown decision")
    if decision == config.DECISION_CLEAR and any(result["breach_codes"] for result in results):
        raise ValueError("CLEAR requires every spot to comply")
    if decision == config.DECISION_CLEAR_WITH_RESTRICTIONS and (
            not any(result["breach_codes"] for result in results)
            or not any(not result["breach_codes"] for result in results)):
        raise ValueError("Decision is not open")
    if decision == config.DECISION_REJECT:
        withdrawn = [result["spot_reference"] for result in results]
        return {
            "spots_cleared": [],
            "spots_withdrawn": withdrawn,
            "spots_cleared_count": 0,
            "spots_withdrawn_count": len(withdrawn),
        }
    cleared = [result["spot_reference"] for result in results if not result["breach_codes"]]
    withdrawn = ([result["spot_reference"] for result in results if result["breach_codes"]]
                 if decision == config.DECISION_CLEAR_WITH_RESTRICTIONS else [])
    return {
        "spots_cleared": cleared,
        "spots_withdrawn": withdrawn,
        "spots_cleared_count": len(cleared),
        "spots_withdrawn_count": len(withdrawn),
    }
