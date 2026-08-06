class PlannerError(Exception):
    pass


class DuplicateTaskError(PlannerError):
    pass


class UnknownDependencyError(PlannerError):
    pass


class DependencyCycleError(PlannerError):
    pass
