"""Модели HTTP-проверки, контекста и строгого результата классификации запросов."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonBlankText = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class _ClassifierModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        hide_input_in_errors=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class HealthResponse(BaseModel):
    """Ответ о работоспособности HTTP API.

    Attributes:
        status: Значение «ok», подтверждающее доступность процесса API.
    """

    status: Literal["ok"] = "ok"


class ContextMessage(_ClassifierModel):
    """Сообщение недавнего контекста, предоставленного серверным кодом.

    Attributes:
        role: Роль пользователя или ассистента; системная роль запрещена.
        content: Непустой текст без изменения исходной формулировки.
    """

    role: Literal["user", "assistant"]
    content: NonBlankText


class PriceRequest(_ClassifierModel):
    """Запрос цены без подстановки её значения из знаний модели.

    Attributes:
        kind: Метка запрошенных сведений price.
    """

    kind: Literal["price"]


class AvailabilityRequest(_ClassifierModel):
    """Запрос наличия товара без утверждения о фактическом остатке.

    Attributes:
        kind: Метка запрошенных сведений availability.
    """

    kind: Literal["availability"]


class FeatureRequest(_ClassifierModel):
    """Запрос конкретной характеристики без утверждения о её наличии.

    Attributes:
        kind: Метка запрошенных сведений feature.
        name: Название характеристики из сообщения или явно связанного контекста.
    """

    kind: Literal["feature"]
    name: NonBlankText


ProductInformationRequest = Annotated[
    PriceRequest | AvailabilityRequest | FeatureRequest, Field(discriminator="kind")
]


class ProductTopic(_ClassifierModel):
    """Вопрос о товаре, его стоимости, наличии или характеристиках.

    Attributes:
        type: Метка темы product.
        product_name: Указанное название либо null, если товар не назван.
        requested_info: Непустой список всех запрошенных сведений о данном товаре.
    """

    type: Literal["product"]
    product_name: NonBlankText | None
    requested_info: list[ProductInformationRequest] = Field(min_length=1)


class DeliveryTopic(_ClassifierModel):
    """Вопрос об общих условиях доставки, без проверки конкретного заказа.

    Attributes:
        type: Метка темы delivery.
        product_name: Названный товар либо null.
    """

    type: Literal["delivery"]
    product_name: NonBlankText | None


class OrderStatusTopic(_ClassifierModel):
    """Вопрос о статусе конкретного заказа без обещания его проверки.

    Attributes:
        type: Метка темы order_status.
        order_number: Явно указанный номер строкой, с сохранением ведущих нулей, либо null.
    """

    type: Literal["order_status"]
    order_number: NonBlankText | None


class ReturnTopic(_ClassifierModel):
    """Запрос обычного возврата с отдельно извлечёнными деталями.

    Attributes:
        type: Метка темы return.
        product_name: Названный товар либо null.
        return_day: Указанный день возврата без домысливания точки отсчёта либо null.
        opened: true, false или null, если о вскрытии не сказано.
        defective: true, false или null, если о браке не сказано.
    """

    type: Literal["return"]
    product_name: NonBlankText | None
    return_day: int | None = Field(ge=0)
    opened: bool | None
    defective: bool | None


class DefectTopic(_ClassifierModel):
    """Обращение по браку без применения правил обычного возврата.

    Attributes:
        type: Метка темы defect.
        product_name: Названный товар либо null.
        return_day: Явно названный день обращения либо null.
        opened: true, false или null, если о вскрытии не сказано.
        defective: true, false или null, если состояние не определено явно.
    """

    type: Literal["defect"]
    product_name: NonBlankText | None
    return_day: int | None = Field(ge=0)
    opened: bool | None
    defective: bool | None


class DiscountTopic(_ClassifierModel):
    """Запрос скидки без решения о возможности её предоставления.

    Attributes:
        type: Метка темы discount.
        product_name: Названный товар либо null.
        requested_percent: Явно запрошенный процент либо null; это не одобрение скидки.
    """

    type: Literal["discount"]
    product_name: NonBlankText | None
    requested_percent: float | None = Field(ge=0)


class OutOfScopeTopic(_ClassifierModel):
    """Отдельная часть запроса за пределами тем исходного регламента.

    Attributes:
        type: Метка темы out_of_scope.
        question: Содержание неизвестной части вопроса без ответа или выдуманных фактов.
    """

    type: Literal["out_of_scope"]
    question: NonBlankText


Topic = Annotated[
    ProductTopic
    | DeliveryTopic
    | OrderStatusTopic
    | ReturnTopic
    | DefectTopic
    | DiscountTopic
    | OutOfScopeTopic,
    Field(discriminator="type"),
]


class ClassificationResult(_ClassifierModel):
    """Проверенная структура классификации, а не бизнес-ответ покупателю.

    Attributes:
        topics: Все темы в порядке запроса с собственными деталями; известные и внешние.
        needs_clarification: Требуется ли уточнение смысла запроса.
        clarification_reason: Причина уточнения либо null, если оно не требуется.
    """

    topics: list[Topic]
    needs_clarification: bool
    clarification_reason: NonBlankText | None

    @model_validator(mode="after")
    def validate_clarification(self) -> Self:
        """Проверить согласованность признака и причины уточнения.

        Returns:
            Результат с согласованным признаком уточнения.

        Raises:
            ValueError: Нет ни тем, ни уточнения, либо причина противоречит признаку.
        """
        if not self.topics and not self.needs_clarification:
            raise ValueError("Без распознанных тем требуется уточнение")
        if self.needs_clarification != (self.clarification_reason is not None):
            raise ValueError("Причина уточнения обязательна только при needs_clarification=true")
        return self
