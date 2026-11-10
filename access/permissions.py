from config import config
def can_access(role,area):return area in config.ROLE_AREA_ACCESS.get(role,[])
def require_area(role,area):
 if not can_access(role,area):raise PermissionError(config.AREA_REFUSED_MESSAGE.format(role=role,area=area))
